"""Survival-oriented Mech Royale bot. Standard library only."""
import math
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
            priorities = {"armor": 100, "plating": 95, "nanites": 92, "servos": 80,
                          "capacitor": 70, "reactor": 65, "jets": 60, "gyros": 55,
                          "gun_rate": 52, "gun_damage": 50, "gun_velocity": 45,
                          "laser_damage": 48, "laser_charge": 46, "mortar_reload": 43,
                          "mortar_splash": 40, "lightning_efficiency": 40,
                          "lightning_range": 35, "sensors": 30}
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
            if dr.get("lands_in", 99) <= 0 and (dr.get("holder") in (None, me.get("id"))):
                d = dist2((x,y), dr["pos"])
                if d < 150 and len(enemies) == 0: loot_candidates.append((d-55, dr["pos"][0], dr["pos"][1], "drop"))
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

        # Zone correction is non-negotiable; keep a strong inward component outside.
        zc = obs["zone"]["center"]
        zr = obs["zone"]["radius"]
        if dist2((x,y), zc) > zr - 12:
            move_x += (zc[0]-x)*2.0
            move_y += (zc[1]-y)*2.0
        mx, my = norm(move_x, move_y)
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
            elif 35 < d < 175 and "mortar" in kinds: preferred = kinds.index("mortar")
            elif 55 < d < 275 and "laser" in kinds: preferred = kinds.index("laser")
            elif "gun" in kinds: preferred = kinds.index("gun")
            elif any(weapons): preferred = next(i for i,w in enumerate(weapons[:2]) if w)
            # Choose a useful weapon and lead its target.
            for i, w in enumerate(weapons[:2]):
                if not w: continue
                kind = w.get("kind")
                stats = w.get("stats", {})
                if kind == "gun":
                    speed = stats.get("speed", 252)
                    rel = [dx, dy, (tp[2]+4.5)-(z+7)]
                    vv = [tv[0], tv[1], tv[2]]
                    it = intercept(rel, vv, speed)
                    qx = dx + tv[0]*it; qy = dy + tv[1]*it; qz = rel[2] + tv[2]*it
                    aim_yaw = math.atan2(qy, qx)
                    aim_pitch = math.atan2(qz, max(1e-6, math.hypot(qx,qy)))
                    act["fire"][i] = (w.get("cooldown", 1) <= 0.06 and w.get("reload", 0) <= 0 and w.get("ammo", 1) > 0)
                    if w.get("ammo", 1) <= 1 and w.get("reload", 0) <= 0: act["reload"] = True
                elif kind == "lightning":
                    aim_yaw = math.atan2(dy,dx)
                    aim_pitch = math.atan2((tp[2]+2)-(z+7), max(1,d))
                    act["fire"][i] = d <= stats.get("range",45)*0.98 and abs(angle_delta(aim_yaw, me.get("aim",[aim_yaw,0])[0])) < 0.5 and w.get("cooldown",1) <= 0.06 and energy >= stats.get("energy",3)
                elif kind == "laser":
                    aim_yaw = math.atan2(dy,dx)
                    aim_pitch = math.atan2((tp[2]+4.5)-(z+7), max(1,d))
                    charge = w.get("charge", 0)
                    # Charge while torso tracks target; release at full charge or if target very close.
                    if charge >= stats.get("charge",0.75)*0.92:
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
