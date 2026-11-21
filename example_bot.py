"""Oval Hunter (v8): kill-oriented Mech Royale bot. Standard library only.

Scoring insight: house bots never fight each other, so points ~= number of other mechs dead when
we die. Kills are what move the score, so v4 engages 1-2 enemies (focus fire on weak targets,
strafe at weapon range, always face the target) and only retreats when badly outnumbered/low HP.
"""
import math
import json
import os
import time
import mechfight as mf

TAU = 2.0 * math.pi

# ---- tunables (flip these between practice submissions to A/B test) ----
AGGRESSION = 1          # 0 = v3-style pure kiting, 1 = engage when odds are acceptable
RETREAT_HP_FRAC = 0.30  # below this fraction of max hp, disengage
MAX_ENGAGE_ENEMIES = 3  # A-bot LLC (10 kills) fought aggressively; v5 used 2
ENGAGE_RANGE = 130.0
TELEMETRY = False
HEAL_ENTER = 0.35       # hp fraction that starts "break contact and heal"
HEAL_EXIT = 0.80
FARM_MAX_CUBES = 28     # top-16 winners finished with ~27 cubes (more upgrade offers); farm drones/crates longer
FARM_RANGE = 90.0


def clamp(x, a, b): return max(a, min(b, x))
def dist2(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def norm(x, y):
    d = math.hypot(x, y)
    return (x/d, y/d) if d > 1e-8 else (0.0, 0.0)
def angle_delta(a, b): return (a-b+math.pi) % TAU - math.pi
def wrap(a): return (a+math.pi) % TAU - math.pi


def intercept(p, v, speed):
    px, py, pz = p
    vx, vy, vz = v
    aa = vx*vx + vy*vy + vz*vz - speed*speed
    bb = 2*(px*vx + py*vy + pz*vz)
    cc = px*px + py*py + pz*pz
    if abs(aa) < 1e-8:
        t = -cc/bb if abs(bb) > 1e-8 else 0.0
    else:
        disc = bb*bb - 4*aa*cc
        if disc < 0:
            t = 0.0
        else:
            r = math.sqrt(disc)
            ts = [q for q in ((-bb-r)/(2*aa), (-bb+r)/(2*aa)) if q > 0]
            t = min(ts) if ts else 0.0
    return clamp(t, 0.0, 1.25)



def mortar_pitch(R, dz, v, g, high=False):
    """Launch pitch to hit a point R m away horizontally and dz m higher. None if out of range."""
    if R < 1.0:
        return None
    disc = v**4 - g*(g*R*R + 2.0*dz*v*v)
    if disc < 0:
        return None
    r = math.sqrt(disc)
    tn = (v*v + r)/(g*R) if high else (v*v - r)/(g*R)
    return math.atan(tn)


COUNTER = {"laser": "lightning", "mortar": "laser", "lightning": "mortar"}  # enemy kind -> best answer


def weapon_value(kind, enemy_kinds):
    v = {"laser": 3.0, "lightning": 2.7, "mortar": 1.0}.get(kind, 0.0)
    for ek in enemy_kinds:
        if COUNTER.get(ek) == kind:
            v += 0.6
    return v


CELL = 24.0


class Bot:
    name = "Oval Hunter v8"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.slot = hello.get("slot", 0)
        self.obstacles = []
        self.grid = {}
        self.target_id = None
        self.strafe_sign = 1.0
        self.strafe_until = 0
        self.last_hp = None
        self.max_ms = 0.0
        self.last_log_t = -99.0
        self.seen_events = 0
        self.kills = 0
        self.last_obs = None
        self.healing = False
        self.cover_cache = (-99, None)
        self._load_map(hello.get("map", {}).get("dir", "/arena"))

    # ------------------------------------------------------------------ map
    def _load_map(self, map_dir):
        world = None
        for path in (os.path.join(map_dir, "v2", "world.json"), "/arena/v2/world.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    world = json.load(f)
                break
            except (OSError, ValueError, TypeError):
                continue
        if not world:
            return
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
            idx = len(self.obstacles)
            self.obstacles.append((min(xs), max(xs), min(ys), max(ys),
                                   float(prism.get("base", -999)),
                                   float(prism.get("top", 999)), pts, holes))
            cx0 = int((min(xs)-1.0) // CELL); cx1 = int((max(xs)+1.0) // CELL)
            cy0 = int((min(ys)-1.0) // CELL); cy1 = int((max(ys)+1.0) // CELL)
            for cx in range(cx0, cx1+1):
                for cy in range(cy0, cy1+1):
                    self.grid.setdefault((cx, cy), []).append(idx)

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
        low, high = feet_z + 1.2, feet_z + 9.0
        for idx in self.grid.get((int(x // CELL), int(y // CELL)), ()):
            x0, x1, y0, y1, base, top, ring, holes = self.obstacles[idx]
            if x < x0-0.8 or x > x1+0.8 or y < y0-0.8 or y > y1+0.8:
                continue
            if top <= low or base >= high:
                continue
            if self._inside_poly(x, y, ring):
                if not any(self._inside_poly(x, y, h) for h in holes):
                    return True
        return False


    def _solid_at(self, x, y, zc):
        for idx in self.grid.get((int(x // CELL), int(y // CELL)), ()):
            x0, x1, y0, y1, base, top, ring, holes = self.obstacles[idx]
            if x < x0 or x > x1 or y < y0 or y > y1:
                continue
            if not (base < zc < top):
                continue
            if self._inside_poly(x, y, ring) and not any(self._inside_poly(x, y, h) for h in holes):
                return True
        return False

    def los_clear(self, ax, ay, az, bx, by, bz):
        """True if no building/bridge prism lies on the straight segment (trees don't block shots)."""
        if not self.obstacles:
            return True
        d = math.hypot(bx-ax, by-ay)
        if d < 4.0:
            return True
        n = int(d / 2.5) + 1
        for k in range(1, n):
            f = k / n
            if self._solid_at(ax+(bx-ax)*f, ay+(by-ay)*f, az+(bz-az)*f):
                return False
        return True

    def find_cover(self, x, y, z, threats, zc, zr):
        """Nearby point hidden from every threat, reachable, and inside the zone."""
        best = None
        for r in (14.0, 26.0, 38.0):
            for k in range(12):
                a = k * TAU / 12.0
                px, py = x + r*math.cos(a), y + r*math.sin(a)
                if math.hypot(px-zc[0], py-zc[1]) > zr - 15.0:
                    continue
                if self._blocked(px, py, z):
                    continue
                hidden = True
                for e in threats:
                    if self.los_clear(e[0], e[1], e[2]+8.0, px, py, z+4.5):
                        hidden = False
                        break
                if not hidden:
                    continue
                ok = True
                for j in range(1, int(r/3.0)+1):
                    f = j*3.0/r
                    if self._blocked(x+(px-x)*f, y+(py-y)*f, z):
                        ok = False
                        break
                if not ok:
                    continue
                mind = min(math.hypot(px-e[0], py-e[1]) for e in threats)
                sc = r - 0.3*min(mind, 80.0)
                if best is None or sc < best[0]:
                    best = (sc, px, py)
            if best:
                break
        return (best[1], best[2]) if best else None

    def navigate(self, x, y, z, gx, gy, vel):
        dx, dy = gx-x, gy-y
        goal_dist = math.hypot(dx, dy)
        if goal_dist < 1.0 or not self.obstacles:
            return norm(dx, dy)
        desired = math.atan2(dy, dx)
        angles = [desired]
        for deg in (15, -15, 30, -30, 45, -45, 60, -60, 80, -80, 105, -105, 130, -130, 155, -155, 180):
            angles.append(desired + math.radians(deg))
        look = min(14.0, max(7.0, goal_dist))
        prev = math.atan2(vel[1], vel[0]) if math.hypot(vel[0], vel[1]) > 1.0 else desired
        best = None
        for a in angles:
            ux, uy = math.cos(a), math.sin(a)
            clearance = look
            blocked = False
            steps = int(look / 1.5) + 1
            for k in range(1, steps+1):
                r = min(look, k*1.5)
                if self._blocked(x+ux*r, y+uy*r, z):
                    clearance = max(0.0, r-1.5)
                    blocked = True
                    break
            end_dist = math.hypot(gx-(x+ux*look), gy-(y+uy*look))
            score = end_dist + 2.2*abs(angle_delta(a, prev)) - 0.22*clearance
            if blocked:
                score += 18.0 + 1.4*(look-clearance)
            if best is None or score < best[0]:
                best = (score, ux, uy)
        return (best[1], best[2]) if best else norm(dx, dy)

    # ----------------------------------------------------------------- zone
    def zone_target(self, obs, x, y):
        zn = obs["zone"]
        cx, cy = zn["center"]
        nx, ny = zn["next_center"]
        radius = zn["radius"]
        t = obs["t"]
        ss, se = zn["shrink_start"], zn["shrink_end"]
        if se > ss and ss <= t <= se:
            f = (t-ss)/(se-ss)
            cx += (nx-cx)*f; cy += (ny-cy)*f
            radius += (zn["next_radius"]-radius)*f
        elif t > se:
            cx, cy, radius = nx, ny, zn["next_radius"]
        d = math.hypot(x-cx, y-cy)
        safe_r = max(0.0, radius - 35.0)
        if d > safe_r or d > max(35.0, safe_r*0.48):
            return (cx, cy)
        return (x + (cx-x)*0.18, y + (cy-y)*0.18)

    # ------------------------------------------------------------ telemetry
    def log_events(self, obs, me):
        if not TELEMETRY:
            return
        t = obs.get("t", 0.0)
        for ev in obs.get("events", []):
            k = ev.get("type")
            if k == "damaged":
                mf.log("t=%.1f DMG %s %.0f%s by=%s hp=%.0f vis=%d" % (
                    t, ev.get("weapon"), ev.get("amount", 0), " REAR" if ev.get("rear") else "",
                    ev.get("by"), me.get("hp", 0), sum(1 for m in obs.get("mechs", []) if m.get("kind") == "mech")))
            elif k == "kill":
                self.kills += 1
                mf.log("t=%.1f KILL target=%s total=%d alive=%s" % (t, ev.get("target"), self.kills, obs.get("alive")))
            elif k in ("upgrade", "took", "effect", "take_failed"):
                mf.log("t=%.1f %s %s" % (t, k, {a: b for a, b in ev.items() if a != "type"}))
        if t - self.last_log_t >= 10.0:
            self.last_log_t = t
            mf.log("t=%.0f STATE hp=%.0f/%.0f alive=%s cubes=%s wpn=%s zone_r=%.0f in_zone=%s max_ms=%.1f" % (
                t, me.get("hp", 0), me.get("max_hp", 0), obs.get("alive"), me.get("cubes"),
                [w.get("kind") if w else None for w in me.get("weapons", [])],
                obs["zone"]["radius"], me.get("in_zone"), self.max_ms))

    # ------------------------------------------------------------------ act
    def act(self, obs):
        t0 = time.perf_counter()
        a = self._act(obs)
        ms = (time.perf_counter()-t0)*1000.0
        self.max_ms = max(self.max_ms, ms)
        return a

    def _act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs.get("t", 0.0)
        tick = obs["tick"]
        hp = me.get("hp", 1400)
        max_hp = max(1.0, me.get("max_hp", 1400))
        energy = me.get("energy", 0)
        vel = me.get("vel", [0, 0, 0])
        mst = me.get("stats", {})
        self.log_events(obs, me)
        mechs_vis = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        enemies = mechs_vis
        npcs = [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]
        act = {"tick": tick, "move": [0.0, 0.0], "boost": False,
               "aim": list(me.get("aim", [0.0, 0.0])), "fire": [False, False]}
        zc = obs["zone"]["center"]; zr = obs["zone"]["radius"]
        hp_frac = hp / max_hp
        if hp_frac < HEAL_ENTER:
            self.healing = True
        elif hp_frac >= HEAL_EXIT:
            self.healing = False

        # ---- upgrades
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            pri = {"armor": 110, "plating": 104, "gun_damage": 100, "gun_rate": 99, "nanites": 96,
                   "servos": 80, "capacitor": 74, "reactor": 72, "laser_damage": 70, "gyros": 66,
                   "jets": 64, "gun_velocity": 62, "sensors": 58, "laser_charge": 52,
                   "mortar_reload": 50, "mortar_splash": 48, "lightning_efficiency": 68,
                   "lightning_range": 60}
            ch = offer["choices"]
            act["pick"] = max(range(len(ch)), key=lambda i: pri.get(ch[i].get("id"), 0))

        weapons = list(me.get("weapons", [None, None]))
        while len(weapons) < 2:
            weapons.append(None)
        kinds = [w.get("kind") if w else None for w in weapons[:2]]
        enemy_kinds = set()
        for e in enemies:
            for k in e.get("weapons", []) or []:
                enemy_kinds.add(k)
        lrange = next((w.get("stats", {}).get("range", 45) for w in weapons[:2] if w and w.get("kind") == "lightning"), 45)

        # ---- choose target: enemy mech (low hp / close / sticky) else a farm target (drone / crate)
        target = None
        tkind = None
        if enemies:
            def score(m):
                d = dist2(me["pos"], m["pos"])
                s = d + 0.07 * m.get("hp", 1000)
                if m["id"] == self.target_id:
                    s -= 35.0
                return s
            target = min(enemies, key=score)
            self.target_id = target["id"]
            tkind = "mech"

        near = [e for e in enemies if dist2((x, y), e["pos"]) < ENGAGE_RANGE]

        # ---- weapon pickups worth getting
        wants = []   # (value, pickup)
        special_vals = [(weapon_value(k, enemy_kinds), i) for i, k in enumerate(kinds) if k and k != "gun"]
        worst_special = min(special_vals) if special_vals else None
        empty_slot = any(w is None for w in weapons[:2])
        for p in obs.get("pickups", []):
            pk = p.get("kind")
            if pk == "gun" or pk in kinds:
                continue
            v = weapon_value(pk, enemy_kinds)
            if empty_slot or (worst_special and v > worst_special[0] + 0.9):
                wants.append((v, p))
        want_weapon = bool(wants) and not near

        # ---- farm target (cubes from drones/crates) when nothing hostile is near
        farm = None
        if (target is None or not near) and not self.healing and me.get("cubes", 0) < FARM_MAX_CUBES \
                and not want_weapon and (kinds[0] == "gun" or "lightning" in kinds):
            best_d = FARM_RANGE
            for n in npcs:
                d = dist2((x, y), n["pos"])
                if d < best_d and self.los_clear(x, y, z+7, n["pos"][0], n["pos"][1], n["pos"][2]+1.8):
                    best_d, farm = d, (n, "npc")
            has_special = any(k and k != "gun" for k in kinds)
            for c in obs.get("crates", []):
                d = dist2((x, y), c["pos"])
                eff = d if has_special else d * 0.55      # crates drop a weapon 50% of the time: hunt them while autocannon-only
                if d < (FARM_RANGE if has_special else 170.0) and eff < best_d \
                        and self.los_clear(x, y, z+7, c["pos"][0], c["pos"][1], c["pos"][2]+1.5):
                    best_d, farm = eff, (c, "crate")
        if farm and not near:
            target, tkind = farm
            target = dict(target); target.setdefault("vel", [0, 0, 0])
        center_off = {"mech": 4.5, "npc": 1.8, "crate": 1.5}.get(tkind, 4.5)

        engage_mech = bool(AGGRESSION and tkind == "mech" and near and len(near) <= MAX_ENGAGE_ENEMIES
                           and not self.healing and dist2((x, y), target["pos"]) < ENGAGE_RANGE)
        engage = engage_mech or (tkind in ("npc", "crate"))

        # ---- weapon choice
        preferred = None
        d_t = 0.0
        if target:
            d_t = dist2((x, y), target["pos"])
            if tkind == "mech":
                if d_t < lrange*0.95 and "lightning" in kinds and energy >= 3: preferred = kinds.index("lightning")
                elif 40 < d_t < 185 and "mortar" in kinds: preferred = kinds.index("mortar")
                elif 40 < d_t < 270 and "laser" in kinds: preferred = kinds.index("laser")
                elif "gun" in kinds: preferred = kinds.index("gun")
                elif any(weapons): preferred = next(i for i, w in enumerate(weapons[:2]) if w)
            else:  # drones / crates: only cheap, safe weapons
                if d_t < lrange*0.95 and "lightning" in kinds and energy >= 12: preferred = kinds.index("lightning")
                elif "gun" in kinds: preferred = kinds.index("gun")
        pref_kind = kinds[preferred] if preferred is not None else None
        desired_range = {"lightning": 0.7*lrange, "mortar": 85.0, "laser": 110.0, "gun": 60.0}.get(pref_kind, 60.0)
        if tkind in ("npc", "crate"):
            desired_range = 0.7*lrange if pref_kind == "lightning" else 35.0

        # ---- loot goal
        zone_goal = self.zone_target(obs, x, y)
        gx, gy = zone_goal
        cands = []
        for c in obs.get("cubes", []):
            d = dist2((x, y), c["pos"])
            if d < 95: cands.append((d-24, c["pos"][0], c["pos"][1]))
        for v, p in wants:
            d = dist2((x, y), p["pos"])
            if d < 90: cands.append((d - 14.0*v, p["pos"][0], p["pos"][1]))
        if not enemies:
            for dr in obs.get("drops", []):
                if dr.get("holder") in (None, me.get("id")):
                    d = dist2((x, y), dr["pos"])
                    tt = d / max(5.0, mst.get("walk_speed", 11.7))
                    if d < 190 and dr.get("lands_in", 99) <= tt + 8.0:
                        cands.append((d-75, dr["pos"][0], dr["pos"][1]))
        if cands and not self.healing:
            b = min(cands, key=lambda q: q[0])
            if b[0] < dist2((x, y), zone_goal) + 15:
                gx, gy = b[1], b[2]

        move_x, move_y = gx-x, gy-y
        hold_still = False

        if engage:
            tp = target["pos"]
            ux, uy = norm(tp[0]-x, tp[1]-y)
            if tick >= self.strafe_until:
                self.strafe_sign = -self.strafe_sign if (tick // 7 + self.slot) % 3 else self.strafe_sign
                self.strafe_until = tick + 14 + (tick * 7 + self.slot) % 12
            radial = clamp((d_t - desired_range) / 18.0, -1.0, 1.0)
            sf = 0.95 if d_t < desired_range + 25.0 else 0.25   # don't spiral when approaching from afar
            move_x = (ux*radial - uy*sf*self.strafe_sign) * 40.0
            move_y = (uy*radial + ux*sf*self.strafe_sign) * 40.0
            for e in near:
                if e is target or e.get("id") == target.get("id"): continue
                dx, dy = x-e["pos"][0], y-e["pos"][1]
                d = max(1.0, math.hypot(dx, dy))
                if d < 40:
                    move_x += dx/d * (40-d); move_y += dy/d * (40-d)
        elif self.healing and enemies:
            # break line of sight and let regeneration run (45 hp/s after 5 s undamaged)
            last_t, cov = self.cover_cache
            if tick - last_t >= 3:
                threats = [e["pos"] for e in enemies]
                cov = self.find_cover(x, y, z, threats, zc, zr)
                self.cover_cache = (tick, cov)
            if cov is not None:
                if math.hypot(cov[0]-x, cov[1]-y) < 3.0:
                    hold_still = True
                move_x, move_y = cov[0]-x, cov[1]-y
            else:
                rx = ry = 0.0
                for e in enemies:
                    dx, dy = x-e["pos"][0], y-e["pos"][1]
                    d = max(1.0, math.hypot(dx, dy))
                    w_ = (1.0 / max(8.0, d)) ** 1.35
                    rx += dx/d*w_; ry += dy/d*w_
                move_x += rx*80; move_y += ry*80
        elif self.healing:
            # nobody in view: stand still (silent) unless the zone forces us out
            if dist2((x, y), zc) < zr - 25:
                hold_still = True
        else:
            if enemies:
                rx = ry = 0.0
                for e in enemies:
                    dx, dy = x-e["pos"][0], y-e["pos"][1]
                    d = max(1.0, math.hypot(dx, dy))
                    w_ = (1.0 / max(8.0, d)) ** 1.35
                    rx += dx/d*w_; ry += dy/d*w_
                nearest = min(dist2((x, y), e["pos"]) for e in enemies)
                if nearest < 58 or len(enemies) >= 3:
                    f = 2.0 if nearest < 30 else 1.2
                    move_x += rx*f*80; move_y += ry*f*80
                for e in enemies:
                    ep = e["pos"]; ev = e.get("vel", [0, 0, 0])
                    ex, ey = ep[0]-x, ep[1]-y
                    rvx, rvy = ev[0]-vel[0], ev[1]-vel[1]
                    vv = rvx*rvx+rvy*rvy
                    tau = clamp(-(ex*rvx+ey*rvy)/vv, 0.0, 1.4) if vv > 1e-6 else 0.0
                    fut = math.hypot(ex+rvx*tau, ey+rvy*tau)
                    if fut < 42.0:
                        aw = norm(x-(ep[0]+ev[0]*tau), y-(ep[1]+ev[1]*tau))
                        s_ = (42.0-fut)/42.0
                        move_x += aw[0]*s_*42.0; move_y += aw[1]*s_*42.0
            else:
                for s_ in obs.get("sounds", []):
                    if s_.get("dist") == "near" and s_.get("kind") in ("gun", "mortar", "laser", "lightning"):
                        bb = s_.get("bearing", 0.0)
                        move_x -= math.cos(bb)*20; move_y -= math.sin(bb)*20

        # ---- projectile dodging
        for pr in obs.get("projectiles", []):
            pp = pr.get("pos", [0, 0, 0]); pv = pr.get("vel", [0, 0, 0])
            rx, ry = x-pp[0], y-pp[1]
            rvx, rvy = vel[0]-pv[0], vel[1]-pv[1]
            vv = rvx*rvx+rvy*rvy
            if vv < 1e-6: continue
            tau = clamp(-(rx*rvx+ry*rvy)/vv, 0.0, 0.8)
            cx, cy = rx+rvx*tau, ry+rvy*tau
            miss = math.hypot(cx, cy)
            if miss < 12.0 and tau > 0.03:
                hold_still = False
                side = 1.0 if (-cy*move_x + cx*move_y) >= 0 else -1.0
                st = (12.0-miss)/12.0 * (1.0-tau/0.9)
                move_x += side*(-rvy)*st*2.0
                move_y += side*(rvx)*st*2.0

        outside = dist2((x, y), zc) > zr - 8
        if dist2((x, y), zc) > zr - 12:
            hold_still = False
            move_x += (zc[0]-x)*2.0; move_y += (zc[1]-y)*2.0

        nx_, ny_ = self.navigate(x, y, z, x+move_x, y+move_y, vel)
        mx, my = norm(nx_, ny_)
        if hold_still:
            mx = my = 0.0
        act["move"] = [mx, my]
        danger = bool(enemies and min(dist2((x, y), e["pos"]) for e in enemies) < 48)
        retreating = bool(enemies) and not engage
        act["boost"] = energy > 18 and ((retreating and danger) or outside or not me.get("in_zone", True))
        act["jump"] = bool(me.get("grounded") and (outside or (retreating and danger and energy > 45)) and tick % 8 == 0)

        # ---- aim & fire
        if target:
            tp = target["pos"]; tv = target.get("vel", [0, 0, 0])
            dx, dy = tp[0]-x, tp[1]-y
            d = math.hypot(dx, dy)
            aim_yaw = math.atan2(dy, dx)
            aim_pitch = 0.0
            cur = me.get("aim", [aim_yaw, 0.0])
            tz = tp[2] + center_off
            los = self.los_clear(x, y, z+7.0, tp[0], tp[1], tz)
            for i, w in enumerate(weapons[:2]):
                if not w or i != preferred:
                    continue
                kind = w.get("kind"); st = w.get("stats", {})
                if kind == "gun":
                    speed = st.get("speed", 252)
                    rel = [dx, dy, tz-(z+7)]
                    it = intercept(rel, tv, speed)
                    qx = dx + tv[0]*it; qy = dy + tv[1]*it; qz = rel[2] + tv[2]*it
                    aim_yaw = math.atan2(qy, qx)
                    aim_pitch = math.atan2(qz, max(1e-6, math.hypot(qx, qy)))
                    yaw_err = abs(angle_delta(aim_yaw, cur[0]))
                    pit_err = abs(aim_pitch - cur[1])
                    tol = 0.09 if d < 90 else 0.05
                    if tkind != "mech": tol = 0.12 if d < 40 else 0.07
                    act["fire"][i] = (los and d < st.get("range", 230) and w.get("cooldown", 1) <= 0.06
                                      and w.get("reload", 0) <= 0 and w.get("ammo", 1) > 0
                                      and yaw_err < tol and pit_err < tol + 0.03)
                    if w.get("ammo", 1) <= 1 and w.get("reload", 0) <= 0:
                        act["reload"] = True
                elif kind == "lightning":
                    aim_pitch = math.atan2(tz-(z+7), max(1, d))
                    act["fire"][i] = (los and d <= st.get("range", 45)*0.98
                                      and abs(angle_delta(aim_yaw, cur[0])) < math.radians(12)
                                      and w.get("cooldown", 1) <= 0.06 and energy >= st.get("energy", 3))
                elif kind == "laser":
                    aim_pitch = math.atan2(tz-(z+7), max(1, d))
                    charge = w.get("charge", 0)
                    aligned = abs(angle_delta(aim_yaw, cur[0])) < 0.08
                    act["fire"][i] = bool(los and aligned and charge < st.get("charge", 0.75)*0.94)
                elif kind == "mortar":
                    speed = st.get("speed", 72); grav = st.get("gravity", 22)
                    px_, py_, pz_ = dx, dy, tz-(z+9.5)
                    th = None
                    for _ in range(3):
                        R = max(1.0, math.hypot(px_, py_))
                        th = mortar_pitch(R, pz_, speed, grav, high=not los)
                        if th is None: break
                        tof = R/(speed*math.cos(th))
                        px_ = dx + tv[0]*tof; py_ = dy + tv[1]*tof; pz_ = tz + tv[2]*tof - (z+9.5)
                    if th is not None:
                        aim_yaw = math.atan2(py_, px_)
                        aim_pitch = th
                        act["fire"][i] = 22 < d < 215 and w.get("cooldown", 1) <= 0.08
            act["aim"] = [wrap(aim_yaw), clamp(aim_pitch, -0.65, 1.25)]
        else:
            if math.hypot(mx, my) > 0.05:
                act["aim"] = [math.atan2(my, mx), 0.0]
            for w in weapons[:2]:
                if w and w.get("kind") == "gun" and w.get("ammo", 12) <= 5 and w.get("reload", 0) <= 0:
                    act["reload"] = True

        # ---- take a pickup (swap out the weakest special weapon if both slots are full)
        pr_ = self.cfg.get("loot", {}).get("pickup_radius", 4) - 0.2
        for v, p in sorted(wants, key=lambda q: -q[0]):
            if abs(p["pos"][2]-z) > 3.8 or dist2((x, y), p["pos"]) > pr_:
                continue
            if weapons[1] is None or weapons[0] is None:
                act["take"] = p["id"]
            elif worst_special:
                slot = worst_special[1]
                w_ = weapons[slot]
                if not (w_ and w_.get("kind") == "laser" and w_.get("charge", 0) > 0):
                    act["take"] = p["id"]; act["drop"] = slot
            break
        return act


mf.run(Bot)
