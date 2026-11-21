"""Survival-oriented Mech Royale bot. Standard library only."""
import math
import json
import os
import mechfight as mf

TAU = 2.0 * math.pi

def clamp(x, a, b): return max(a, min(b, x))
def dist2(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def norm(x, y):
    d = math.hypot(x, y)
    return (x/d, y/d) if d > 1e-8 else (0.0, 0.0)
def angle_delta(a, b): return (a-b+math.pi) % TAU - math.pi
def wrap(a): return (a+math.pi) % TAU - math.pi

def intercept(p, v, speed):
    # Solve |p + vt| = speed*t; robustly clamp to weapon lifetime/range.
    px, py, pz = p
    vx, vy, vz = v
    aa = vx*vx + vy*vy + vz*vz - speed*speed
    bb = 2*(px*vx + py*vy + pz*vz)
    cc = px*px + py*py + pz*pz
    if abs(aa) < 1e-8:
        t = -cc/bb if abs(bb) > 1e-8 else 0.0
    else:
        disc = bb*bb - 4*aa*cc
        if disc < 0: t = 0.0
        else:
            r = math.sqrt(disc)
            ts = [q for q in ((-bb-r)/(2*aa), (-bb+r)/(2*aa)) if q > 0]
            t = min(ts) if ts else 0.0
    return clamp(t, 0.0, 1.25)

class Bot:
    name = "Oval Survivor"
    def __init__(self, hello):
        self.cfg = hello["config"]
        self.slot = hello.get("slot", 0)
        self.last_goal = None
        self.last_enemy = None
        self.last_seen_t = -100.0
        self.laser_release = [False, False]
        self.last_pos = None
        self.obstacles = []
        self.map_bounds = None
        self._load_map(hello.get("map", {}).get("dir", "/arena"))

    def _load_map(self, map_dir):
        """Load static solid footprints once; the match map is read-only."""
        candidates = [os.path.join(map_dir, "v2", "world.json"),
                      "/arena/v2/world.json",
                      "/mnt/data/mechkit/mechfight-kit/arena/v2/world.json"]
        world = None
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    world = json.load(f)
                break
            except (OSError, ValueError, TypeError):
                continue
        if not world:
            return
        self.map_bounds = world.get("bounds")
        for prism in world.get("prisms", []):
            ring = prism.get("ring", [])
            if len(ring) < 6:
                continue
            pts = [(ring[i], ring[i+1]) for i in range(0, len(ring)-1, 2)]
            xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
            holes = []
            for h in prism.get("holes", []):
                if len(h) >= 6:
                    holes.append([(h[i], h[i+1]) for i in range(0, len(h)-1, 2)])
            self.obstacles.append((min(xs), max(xs), min(ys), max(ys),
                                   float(prism.get("base", -999)),
                                   float(prism.get("top", 999)), pts, holes))

    @staticmethod
    def _inside_poly(x, y, poly):
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and x < (xj-xi)*(y-yi)/(yj-yi+1e-12)+xi:
                inside = not inside
            j = i
        return inside

    def _blocked(self, x, y, feet_z):
        # A prism blocks a walking path if it intersects the mech's 9 m body.
        # Bridges with >=9 m clearance remain traversable; holes remain open.
        low, high = feet_z + 1.2, feet_z + 9.0
        for x0, x1, y0, y1, base, top, ring, holes in self.obstacles:
            if x < x0-0.8 or x > x1+0.8 or y < y0-0.8 or y > y1+0.8:
                continue
            if top <= low or base >= high:
                continue
            if self._inside_poly(x, y, ring):
                if not any(self._inside_poly(x, y, h) for h in holes):
                    return True
        return False

    def navigate(self, x, y, z, gx, gy, current_vel):
        """Short-horizon sampled heading planner over the real building footprints."""
        dx, dy = gx-x, gy-y
        goal_dist = math.hypot(dx, dy)
        if goal_dist < 1.0 or not self.obstacles:
            return norm(dx, dy)
        desired = math.atan2(dy, dx)
        # Include exact goal bearing and a fan of alternatives; evaluate 14 m ahead.
        angles = [desired]
        for deg in (15, -15, 30, -30, 45, -45, 60, -60, 80, -80,
                    105, -105, 130, -130, 155, -155, 180):
            angles.append(desired + math.radians(deg))
        look = min(14.0, max(7.0, goal_dist))
        best = None
        prev = math.atan2(current_vel[1], current_vel[0]) if math.hypot(current_vel[0], current_vel[1]) > 1.0 else desired
        for a in angles:
            ux, uy = math.cos(a), math.sin(a)
            clearance = look
            blocked = False
            # Sample at 1.5 m spacing with a small lateral footprint margin.
            steps = int(look / 1.5) + 1
            for k in range(1, steps+1):
                r = min(look, k*1.5)
                qx, qy = x+ux*r, y+uy*r
                if self._blocked(qx, qy, z):
                    clearance = max(0.0, r-1.5)
                    blocked = True
                    break
            # Score progress, path clearance, and smoothness. Penalize dead ends strongly.
            end_dist = math.hypot(gx-(x+ux*look), gy-(y+uy*look))
            turn = abs(angle_delta(a, prev))
            score = (end_dist + 2.2*turn - 0.22*clearance)
            if blocked:
                score += 18.0 + 1.4*(look-clearance)
            if best is None or score < best[0]:
                best = (score, ux, uy)
        return (best[1], best[2]) if best else norm(dx,dy)

    def zone_target(self, obs, x, y):
        z = obs["zone"]
        cx, cy = z["center"]
        nx, ny = z["next_center"]
        radius = z["radius"]
        t = obs["t"]
        ss, se = z["shrink_start"], z["shrink_end"]
        if se > ss and ss <= t <= se:
            f = (t-ss)/(se-ss)
            cx += (nx-cx)*f; cy += (ny-cy)*f
            radius += (z["next_radius"]-radius)*f
        elif t > se:
            cx, cy, radius = nx, ny, z["next_radius"]
        dx, dy = x-cx, y-cy
        d = math.hypot(dx, dy)
        # Stay comfortably inside, but avoid blindly running to exact center.
        safe_r = max(0.0, radius - 35.0)
        if d > safe_r:
            return (cx, cy)
        # Drift toward the inner half of the circle rather than camping at its edge.
        if d > max(35.0, safe_r * 0.48):
            return (cx, cy)
        return (x + (cx-x)*0.18, y + (cy-y)*0.18)

    def act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs.get("t", 0.0)
        hp = me.get("hp", 1400)
        energy = me.get("energy", 0)
        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        npcs = [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]
        act = {"tick": obs["tick"], "move": [0.0, 0.0], "boost": False,
               "aim": list(me.get("aim", [0.0, 0.0])), "fire": [False, False]}

        # Upgrade choices: survival first, then mobility/weapon quality.
        offer = obs.get("offer")
        if offer:
            priorities = {"armor": 110, "plating": 102, "nanites": 98, "servos": 82,
                          "capacitor": 74, "reactor": 72, "jets": 68, "gyros": 62,
                          "sensors": 58, "gun_rate": 54, "gun_damage": 53, "gun_velocity": 48,
                          "laser_damage": 52, "laser_charge": 49, "mortar_reload": 46,
                          "mortar_splash": 44, "lightning_efficiency": 44,
                          "lightning_range": 40}
            choices = offer.get("choices", [])
            if choices:
                act["pick"] = max(range(len(choices)), key=lambda i: priorities.get(choices[i].get("id"), 0))

        # Select target: nearest visible mech, but favor damaged/nearby threats.
        target = None
        if enemies:
            def threat(m):
                d = dist2(me["pos"], m["pos"])
                return d + 0.025 * m.get("hp", 1000)
            target = min(enemies, key=threat)
            self.last_enemy = target["id"]
            self.last_seen_t = t

        # Loot goals: cubes and weapon pickups, weighted by distance and value.
        zone_goal = self.zone_target(obs, x, y)
        gx, gy = zone_goal
        loot_candidates = []
        for c in obs.get("cubes", []):
            d = dist2((x,y), c["pos"])
            if d < 95: loot_candidates.append((d - 24, c["pos"][0], c["pos"][1], "cube"))
        for p in obs.get("pickups", []):
            if p.get("kind") == "gun": continue
            kinds = [w.get("kind") for w in me.get("weapons", []) if w]
            # Weapon diversity: strongly value the first useful special weapon.
            if p.get("kind") in kinds: continue
            d = dist2((x,y), p["pos"])
            if d < 75: loot_candidates.append((d - (32 if not any(k in kinds for k in ("mortar","laser","lightning")) else 10), p["pos"][0], p["pos"][1], "weapon"))
        for dr in obs.get("drops", []):
            if dr.get("holder") in (None, me.get("id")):
                d = dist2((x,y), dr["pos"])
                # Start approaching while it falls; arrive shortly after landing and capture only if uncontested.
                travel_time = d / max(5.0, me.get("stats",{}).get("walk_speed",11.7))
                if d < 190 and len(enemies) == 0 and dr.get("lands_in",99) <= travel_time + 8.0:
                    loot_candidates.append((d-75, dr["pos"][0], dr["pos"][1], "drop"))
        if loot_candidates:
            best = min(loot_candidates, key=lambda q:q[0])
            # Loot only if it does not require leaving the safe circle.
            if best[0] < dist2((x,y), zone_goal) + 15:
                gx, gy = best[1], best[2]

        # Combat posture: kite threats, don't charge into a squad.
        move_x, move_y = gx-x, gy-y
        if enemies:
            # Repel away from every nearby mech; strongest at close range.
            rx = ry = 0.0
            for e in enemies:
                dx, dy = x-e["pos"][0], y-e["pos"][1]
                d = max(1.0, math.hypot(dx,dy))
                weight = (1.0 / max(8.0, d)) ** 1.35
                rx += dx/d * weight
                ry += dy/d * weight
            nearest = min(dist2((x,y), e["pos"]) for e in enemies)
            # Retreat when close or outnumbered; otherwise continue rotating/looting.
            if nearest < 58 or len(enemies) >= 3:
                factor = 2.0 if nearest < 30 else 1.2
                move_x += rx * factor * 80
                move_y += ry * factor * 80
        # Sounds from rear: move away from the source bearing if no visible target.
        if not enemies:
            for s in obs.get("sounds", []):
                if s.get("dist") == "near" and s.get("kind") in ("gun", "mortar", "laser", "lightning"):
                    b = s.get("bearing", 0.0)
                    move_x -= math.cos(b)*20
                    move_y -= math.sin(b)*20

        # Predict incoming projectiles and sidestep their future paths (velocity-obstacle idea).
        # Projectiles are observed with world-space velocity; avoid their closest approach over 0.8 s.
        for pr in obs.get("projectiles", []):
            pp = pr.get("pos", [0,0,0]); pv = pr.get("vel", [0,0,0])
            rx, ry = x-pp[0], y-pp[1]
            vx, vy = me.get("vel", [0,0,0])[:2]
            rvx, rvy = vx-pv[0], vy-pv[1]
            vv = rvx*rvx+rvy*rvy
            if vv < 1e-6: continue
            tau = clamp(-(rx*rvx+ry*rvy)/vv, 0.0, 0.8)
            cx, cy = rx+rvx*tau, ry+rvy*tau
            miss = math.hypot(cx,cy)
            if miss < 12.0 and tau > 0.03:
                # Perpendicular dodge, choosing the side that makes progress toward the goal.
                side = 1.0 if (-(cy)*move_x + cx*move_y) >= 0 else -1.0
                strength = (12.0-miss)/12.0 * (1.0-tau/0.9)
                move_x += side*(-rvy)*strength*2.0
                move_y += side*( rvx)*strength*2.0

        # Avoid predicted enemy interception, not merely current proximity.
        # This is a lightweight velocity-obstacle approximation suitable for the 50 ms budget.
        for e in enemies:
            ep=e.get("pos",[0,0,0]); ev=e.get("vel",[0,0,0])
            ex,ey=ep[0]-x,ep[1]-y
            rvx=ev[0]-me.get("vel",[0,0,0])[0]; rvy=ev[1]-me.get("vel",[0,0,0])[1]
            vv=rvx*rvx+rvy*rvy
            tau=clamp(-(ex*rvx+ey*rvy)/vv,0.0,1.4) if vv>1e-6 else 0.0
            future=math.hypot(ex+rvx*tau,ey+rvy*tau)
            if future < 42.0:
                away=norm(x-(ep[0]+ev[0]*tau), y-(ep[1]+ev[1]*tau))
                strength=(42.0-future)/42.0
                move_x += away[0]*strength*42.0
                move_y += away[1]*strength*42.0

        # Zone correction is non-negotiable; keep a strong inward component outside.
        zc = obs["zone"]["center"]
        zr = obs["zone"]["radius"]
        if dist2((x,y), zc) > zr - 12:
            move_x += (zc[0]-x)*2.0
            move_y += (zc[1]-y)*2.0
        # Replace direct steering with map-aware local path selection.
        nav_x, nav_y = self.navigate(x, y, z, x+move_x, y+move_y, me.get("vel", [0,0,0]))
        mx, my = norm(nav_x, nav_y)
        act["move"] = [mx, my]
        # Spend boost to escape threats / zone, otherwise conserve for emergencies.
        danger = bool(enemies and min(dist2((x,y), e["pos"]) for e in enemies) < 48)
        outside = dist2((x,y), zc) > zr - 8
        act["boost"] = energy > 18 and (danger or outside or (not me.get("in_zone", True)))
        act["jump"] = bool(me.get("grounded") and (outside or (danger and energy > 45)) and obs["tick"] % 8 == 0)

        # Face an attacker whenever possible to avoid the 1.5x rear damage penalty.
        if target:
            tp = target["pos"]; tv = target.get("vel", [0,0,0])
            dx, dy = tp[0]-x, tp[1]-y
            d = math.hypot(dx,dy)
            weapons = me.get("weapons", [None, None])
            if len(weapons) < 2: weapons = list(weapons) + [None]*(2-len(weapons))
            aim_yaw = math.atan2(dy, dx)
            # Pick one aiming solution for the weapon best suited to this range.
            kinds = [w.get("kind") if w else None for w in weapons[:2]]
            preferred = None
            if d < 43 and "lightning" in kinds: preferred = kinds.index("lightning")
            elif 38 < d < 185 and "mortar" in kinds: preferred = kinds.index("mortar")
            elif 65 < d < 270 and "laser" in kinds: preferred = kinds.index("laser")
            elif "gun" in kinds: preferred = kinds.index("gun")
            elif any(weapons): preferred = next(i for i,w in enumerate(weapons[:2]) if w)
            # Choose a single weapon and compute its own aiming solution.
            # Firing both slots while aiming only one causes systematic misses.
            for i, w in enumerate(weapons[:2]):
                if not w: continue
                kind = w.get("kind")
                if i != preferred:
                    act["fire"][i] = False
                    continue
                stats = w.get("stats", {})
                if kind == "gun":
                    speed = stats.get("speed", 252)
                    rel = [dx, dy, (tp[2]+4.5)-(z+7)]
                    vv = [tv[0], tv[1], tv[2]]
                    it = intercept(rel, vv, speed)
                    qx = dx + tv[0]*it; qy = dy + tv[1]*it; qz = rel[2] + tv[2]*it
                    aim_yaw = math.atan2(qy, qx)
                    aim_pitch = math.atan2(qz, max(1e-6, math.hypot(qx,qy)))
                    yaw_err = abs(angle_delta(aim_yaw, me.get("aim",[aim_yaw,0])[0]))
                    pitch_err = abs(aim_pitch - me.get("aim",[0,aim_pitch])[1])
                    act["fire"][i] = (w.get("cooldown", 1) <= 0.06 and w.get("reload", 0) <= 0 and w.get("ammo", 1) > 0 and yaw_err < 0.10 and pitch_err < 0.10)
                    if w.get("ammo", 1) <= 1 and w.get("reload", 0) <= 0: act["reload"] = True
                elif kind == "lightning":
                    aim_yaw = math.atan2(dy,dx)
                    aim_pitch = math.atan2((tp[2]+2)-(z+7), max(1,d))
                    act["fire"][i] = d <= stats.get("range",45)*0.98 and abs(angle_delta(aim_yaw, me.get("aim",[aim_yaw,0])[0])) < math.radians(12) and w.get("cooldown",1) <= 0.06 and energy >= stats.get("energy",3)
                elif kind == "laser":
                    aim_yaw = math.atan2(dy,dx)
                    aim_pitch = math.atan2((tp[2]+4.5)-(z+7), max(1,d))
                    charge = w.get("charge", 0)
                    # Charge while torso tracks target; release at full charge or if target very close.
                    if charge >= stats.get("charge",0.75)*0.94:
                        act["fire"][i] = False
                    else:
                        act["fire"][i] = True
                elif kind == "mortar":
                    # Approximate ballistic pitch for a moving target; avoid self-splash up close.
                    speed = stats.get("speed",72); grav = stats.get("gravity",22)
                    it = clamp(d/max(20,speed), 0.15, 2.2)
                    tx = dx + tv[0]*it; ty = dy + tv[1]*it
                    horizontal = max(1, math.hypot(tx,ty))
                    dz = (tp[2]+2.0)-(z+9.5) + 0.5*grav*it*it
                    aim_yaw = math.atan2(ty,tx)
                    aim_pitch = math.atan2(dz,horizontal)
                    act["fire"][i] = d > 16 and d < 190 and w.get("cooldown",1) <= 0.08
                # Latest weapon slot may override aim; slot 0 preferred if gun is present.
                if i == preferred:
                    act["aim"] = [wrap(aim_yaw), clamp(aim_pitch, -0.65, 1.25)]
            # If no weapon is available, still face the nearest attacker for front armor.
            if preferred is None:
                act["aim"] = [wrap(math.atan2(dy,dx)), clamp(math.atan2((tp[2]+4.5)-(z+7),max(1,d)),-0.65,1.25)]
            # Fire gun/other slot only if it has a viable target; no blind firing.
        else:
            # Scan toward nearest sound; otherwise face travel direction.
            if math.hypot(mx,my) > 0.05:
                act["aim"] = [math.atan2(my,mx), 0.0]
            for i,w in enumerate(me.get("weapons",[])[:2]):
                if w and w.get("kind") == "gun" and w.get("ammo",0) <= 2 and w.get("reload",0) <= 0:
                    act["reload"] = True

        # Take nearby special weapon if there is an empty slot.
        weapons = me.get("weapons", [None,None])
        if len(weapons) < 2: weapons = list(weapons)+[None]*(2-len(weapons))
        for p in obs.get("pickups", []):
            if p.get("kind") == "gun": continue
            if any(w and w.get("kind") == p.get("kind") for w in weapons): continue
            if dist2((x,y),p["pos"]) <= self.cfg.get("loot",{}).get("pickup_radius",4)-0.15:
                if weapons[1] is None:
                    act["take"] = p["id"]
                elif weapons[0] is None:
                    act["take"] = p["id"]
                break

        # Reload before empty where safe.
        for w in weapons:
            if w and w.get("kind") == "gun" and w.get("ammo",12) <= 3 and w.get("reload",0) <= 0 and not enemies:
                act["reload"] = True

        return act

mf.run(Bot)
