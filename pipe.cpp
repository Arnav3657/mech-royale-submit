// Shopping Sessions pipeline. Single static binary: parses the CSVs with mmap + AVX2,
// builds both tables, and writes Parquet directly (no Python / DuckDB step).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

typedef int64_t i64;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int32_t i32_t;
typedef std::string_view sv;

static const i64 DAY_MS = 86400000LL;
static const i64 WEEK_MS = 7 * DAY_MS;
static const i64 BADTS = INT64_MIN;

static int NT = 8;
static bool PROF = false;
static double T0 = 0;
static double nowms() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
#include <sys/resource.h>
#define TICK(lbl) do { if (PROF) { double t = nowms(); struct rusage ru; getrusage(RUSAGE_SELF, &ru); fprintf(stderr, "%-12s %8.1f ms  (user %.0f sys %.0f ms, minflt %ld)\n", lbl, t - T0, ru.ru_utime.tv_sec*1e3+ru.ru_utime.tv_usec/1e3, ru.ru_stime.tv_sec*1e3+ru.ru_stime.tv_usec/1e3, ru.ru_minflt); T0 = t; } } while (0)
static std::string DATA, OUT, SCR;

// ---------------------------------------------------------------- hashing
static inline u64 rd8(const char* p) { u64 v; memcpy(&v, p, 8); return v; }
static inline u64 mix(u64 a, u64 b) {
  __uint128_t r = (__uint128_t)a * b;
  return (u64)r ^ (u64)(r >> 64);
}
static inline u64 hashb(const char* p, size_t n) {
  const u64 k0 = 0xa0761d6478bd642fULL, k1 = 0xe7037ed1a0b428dbULL, k2 = 0x8ebc6af09c88c6e3ULL;
  u64 h = k0 ^ (n * k2);
  while (n >= 8) { h = mix(h ^ rd8(p), k1); p += 8; n -= 8; }
  if (n) {
    u64 t = 0;
    memcpy(&t, p, n);
    h = mix(h ^ t ^ (n << 56), k1);
  }
  return mix(h, k2 ^ 0x9e3779b97f4a7c15ULL);
}

// ---------------------------------------------------------------- files
struct MFile { const char* p; size_t n; };
static std::vector<std::string> listcsv(const std::string& dir) {
  std::vector<std::string> v;
  DIR* d = opendir(dir.c_str());
  if (!d) return v;
  while (dirent* e = readdir(d)) {
    std::string nm = e->d_name;
    if (nm.size() >= 4 && nm.compare(nm.size() - 4, 4, ".csv") == 0) v.push_back(dir + "/" + nm);
  }
  closedir(d);
  std::sort(v.begin(), v.end());
  return v;
}
static MFile mapfile(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return {nullptr, 0};
  struct stat st;
  fstat(fd, &st);
  size_t n = st.st_size;
  if (n == 0) { close(fd); return {nullptr, 0}; }
  void* p = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return {nullptr, 0};
  madvise(p, n, MADV_SEQUENTIAL);
  return {(const char*)p, n};
}

struct Task { int file; size_t s, e; };
static std::vector<Task> mktasks(const std::vector<MFile>& fs, size_t chunk) {
  std::vector<Task> t;
  for (size_t i = 0; i < fs.size(); i++) {
    size_t n = fs[i].n;
    if (!n) continue;
    for (size_t s = 0; s < n; s += chunk) t.push_back({(int)i, s, std::min(n, s + chunk)});
  }
  return t;
}
// iterate over lines belonging to the task (excluding header line)
template <class F>
static inline void for_lines(const MFile& f, const Task& t, F fn) {
  const size_t safeLimit = f.n >= 320 ? f.n - 320 : 0;
  const char* b = f.p;
  size_t pos = t.s;
  if (pos == 0) {
    const char* nl = (const char*)memchr(b, '\n', f.n);
    if (!nl) return;
    pos = nl - b + 1;
  } else if (b[pos - 1] != '\n') {
    const char* nl = (const char*)memchr(b + pos, '\n', f.n - pos);
    if (!nl) return;
    pos = nl - b + 1;
  }
  while (pos < t.e && pos < f.n) {
    const char* ls = b + pos;
    const char* nl = (const char*)memchr(ls, '\n', f.n - pos);
    size_t len = nl ? (size_t)(nl - ls) : f.n - pos;
    fn(ls, len, pos <= safeLimit);
    pos += len + 1;
  }
}

// split line into exactly K fields (unquoted values). false => malformed
template <int K>
static inline bool split_scalar(const char* p, size_t len, sv* out) {
  const char* end = p + len;
  for (int i = 0; i < K; i++) {
    const char* c = (i == K - 1) ? nullptr : (const char*)memchr(p, ',', end - p);
    const char* fe;
    if (i == K - 1) {
      if (memchr(p, ',', end - p)) return false;
      fe = end;
    } else {
      if (!c) return false;
      fe = c;
    }
    size_t n = fe - p;
    if (n && p[0] == '"') {
      if (n < 2 || fe[-1] != '"') return false;
      if (n > 2 && memchr(p + 1, '"', n - 2)) return false;
      out[i] = sv(p + 1, n - 2);
    } else {
      if (n && memchr(p, '"', n)) return false;
      out[i] = sv(p, n);
    }
    if (i < K - 1) p = fe + 1;
  }
  return true;
}
#include <immintrin.h>
// AVX2 splitter: needs 32 readable bytes past the line start block; caller guarantees slack via `safe`.
template <int K>
static inline bool split(const char* p, size_t len, sv* out, bool safe = false) {
  if (!safe || len > 255) return split_scalar<K>(p, len, out);
  u64 cm[4] = {0, 0, 0, 0}, qm[4] = {0, 0, 0, 0};
  const __m256i vc = _mm256_set1_epi8(','), vq = _mm256_set1_epi8('"');
  size_t nb = (len + 63) >> 6;
  for (size_t b = 0; b < nb; b++) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(p + b * 64));
    __m256i a1 = _mm256_loadu_si256((const __m256i*)(p + b * 64 + 32));
    u64 c = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a0, vc)) | ((u64)(u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a1, vc)) << 32);
    u64 q = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a0, vq)) | ((u64)(u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a1, vq)) << 32);
    cm[b] = c; qm[b] = q;
  }
  size_t rem = len & 63;
  if (rem) { u64 keep = (1ULL << rem) - 1; cm[nb - 1] &= keep; qm[nb - 1] &= keep; }
  int ncomma = 0, nq = 0;
  for (size_t b = 0; b < nb; b++) { ncomma += __builtin_popcountll(cm[b]); nq += __builtin_popcountll(qm[b]); }
  if (ncomma != K - 1) return false;
  size_t start = 0;
  int fi = 0;
  int expectq = 0;
  for (size_t b = 0; b < nb; b++) {
    u64 m = cm[b];
    while (m) {
      size_t pos = b * 64 + __builtin_ctzll(m);
      m &= m - 1;
      size_t n = pos - start;
      if (n && p[start] == '"') {
        if (n < 2 || p[pos - 1] != '"') return false;
        out[fi] = sv(p + start + 1, n - 2); expectq += 2;
      } else out[fi] = sv(p + start, n);
      fi++; start = pos + 1;
    }
  }
  size_t n = len - start;
  if (n && p[start] == '"') {
    if (n < 2 || p[len - 1] != '"') return false;
    out[fi] = sv(p + start + 1, n - 2); expectq += 2;
  } else out[fi] = sv(p + start, n);
  return nq == expectq;
}

// ---------------------------------------------------------------- time
static inline i64 days_from_civil(i64 y, unsigned m, unsigned d) {
  y -= m <= 2;
  i64 era = (y >= 0 ? y : y - 399) / 400;
  unsigned yoe = (unsigned)(y - era * 400);
  unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (i64)doe - 719468;
}
static inline bool isdig(char c) { return c >= '0' && c <= '9'; }
static inline int d2(const char* p) { return (p[0] - '0') * 10 + (p[1] - '0'); }
static inline bool d2ok(const char* p) { return isdig(p[0]) && isdig(p[1]); }
static i64 parse_ts(sv s) {
  size_t n = s.size();
  const char* p = s.data();
  if (n == 0) return BADTS;
  if (n == 20 || n == 25) {
    if (!(isdig(p[0]) && isdig(p[1]) && isdig(p[2]) && isdig(p[3]))) goto f3;
    if (p[4] != '-' || p[7] != '-' || p[10] != 'T' || p[13] != ':' || p[16] != ':') return BADTS;
    if (!d2ok(p + 5) || !d2ok(p + 8) || !d2ok(p + 11) || !d2ok(p + 14) || !d2ok(p + 17)) return BADTS;
    i64 off = 0;
    if (n == 20) {
      if (p[19] != 'Z') return BADTS;
    } else {
      if (p[19] != '+' && p[19] != '-') return BADTS;
      if (p[22] != ':' || !d2ok(p + 20) || !d2ok(p + 23)) return BADTS;
      int oh = d2(p + 20), om = d2(p + 23);
      if (oh > 23 || om > 59) return BADTS;
      off = (oh * 3600 + om * 60) * 1000LL;
      if (p[19] == '-') off = -off;
    }
    int y = (p[0] - '0') * 1000 + (p[1] - '0') * 100 + (p[2] - '0') * 10 + (p[3] - '0');
    int mo = d2(p + 5), dd = d2(p + 8), hh = d2(p + 11), mi = d2(p + 14), ss = d2(p + 17);
    if (mo < 1 || mo > 12 || dd < 1 || hh > 23 || mi > 59 || ss > 59) return BADTS;
    static const int dim[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int dm = dim[mo - 1];
    if (mo == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) dm = 29;
    if (dd > dm) return BADTS;
    i64 days = days_from_civil(y, mo, dd);
    return ((days * 86400LL) + hh * 3600 + mi * 60 + ss) * 1000LL - off;
  }
f3:
  {
    u64 v = 0;
    for (size_t i = 0; i < n; i++) {
      if (!isdig(p[i])) return BADTS;
      u64 dgt = p[i] - '0';
      if (v > (u64)(INT64_MAX - dgt) / 10) return BADTS;
      v = v * 10 + dgt;
    }
    return (i64)v;
  }
}

// ---------------------------------------------------------------- tables
struct HT {  // hash64 -> u32 value (open addressing)
  std::vector<u64> keys;
  std::vector<u32> vals;
  u64 mask = 0;
  void init(size_t n) {
    size_t c = 16;
    while (c < n * 2) c <<= 1;
    keys.assign(c, 0);
    vals.assign(c, 0xFFFFFFFFu);
    mask = c - 1;
  }
  static inline u64 fix(u64 h) { return h ? h : 1; }
  u32 get(u64 h) const {
    h = fix(h);
    u64 i = h & mask;
    while (true) {
      if (keys[i] == h) return vals[i];
      if (keys[i] == 0) return 0xFFFFFFFFu;
      i = (i + 1) & mask;
    }
  }
  // returns existing value or inserts v
  u32 getput(u64 h, u32 v, bool& inserted) {
    h = fix(h);
    u64 i = h & mask;
    while (true) {
      if (keys[i] == h) { inserted = false; return vals[i]; }
      if (keys[i] == 0) { keys[i] = h; vals[i] = v; inserted = true; return v; }
      i = (i + 1) & mask;
    }
  }
};

// links / persons
static HT linkHT;
static HT prodHT;
static std::vector<u16> prodCat;
static HT campHT;
static std::vector<std::string> campId, campChan;
static std::vector<std::string> catNames;  // by rank
static u16 UNKCAT;
static i64 WS, WE, DAY0;
static int NDAYS;

struct Row2 { sv a, b; };

// ---------------------------------------------------------------- records
struct ERaw {  // pre-dedup, user/product still unresolved
  u64 eid, uh, unm, ph;
  i64 ts;
  u32 meta;  // price24 | type<<24 | ok<<26
  u32 ing;   // ingest_ts in whole seconds, saturated
};
struct CRaw {
  u64 uh;
  i64 ts;
  u64 cid, ch, unm;
};
struct ERec {  // post-dedup
  u64 pkey;
  i64 ts;
  u64 nm;
  u32 meta;  // price | type<<24
  u16 cat;
  u16 pad;
};
struct CRec {
  u64 pkey;
  i64 ts;
  const char* cid;
  u32 cidlen;
  u32 camp;
};

static const int RING = 16;
static const int NPART = 256;   // dedup partitions
static const int NBUCK = 1024;  // person buckets
static inline u64 mkNm(const char* p, size_t n) { return (u64)(uintptr_t)p | ((u64)n << 48); }
static inline sv getNm(u64 v) { return sv((const char*)(uintptr_t)(v & 0xFFFFFFFFFFFFULL), v >> 48); }
static inline int bucketOf(u64 pk) { return (int)((pk >> 20) & (NBUCK - 1)); }
static inline int partOf(u64 h) { return (int)((h >> 24) & (NPART - 1)); }

template <class F>
static void run_threads(int n, F fn) {
  std::vector<std::thread> th;
  for (int i = 0; i < n; i++) th.emplace_back([=]() { fn(i); });
  for (auto& t : th) t.join();
}

// resolve user -> (pkey, nm)
struct LSlot { u64 h, pk, nm, pad; };
static std::vector<LSlot> lTab;
static u64 lMask = 0;
static inline void prefetchUser(u64 h) {
  if (lMask) __builtin_prefetch(&lTab[(h ? h : 1) & lMask]);
}
static inline void resolveUserH(sv u, u64 h, u64& pk, u64& nm) {
  u64 hh = h ? h : 1;
  if (lMask) {
    u64 i = hh & lMask;
    while (true) {
      const LSlot& sl = lTab[i];
      if (sl.h == hh) { pk = sl.pk; nm = sl.nm; return; }
      if (sl.h == 0) break;
      i = (i + 1) & lMask;
    }
  }
  pk = h;
  nm = mkNm(u.data(), u.size());
}
static inline void resolveUser(sv u, u64& pk, u64& nm) { resolveUserH(u, hashb(u.data(), u.size()), pk, nm); }

static std::vector<MFile> readAll(const std::string& dir, std::vector<std::string>& names) {
  names = listcsv(dir);
  std::vector<MFile> fs;
  for (auto& n : names) fs.push_back(mapfile(n));
  return fs;
}

static bool lessSv(sv a, sv b) { return a.compare(b) < 0; }

// simple json value extraction
static std::string jget(const std::string& s, const char* key) {
  size_t k = s.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) return "";
  size_t c = s.find(':', k);
  size_t q1 = s.find('"', c + 1);
  size_t q2 = s.find('"', q1 + 1);
  return s.substr(q1 + 1, q2 - q1 - 1);
}

// escape names for text output
static inline void putName(std::string& o, sv s) {
  bool plain = true;
  for (unsigned char c : s) if (c < 0x20 || c > 0x7e) { plain = false; break; }
  if (plain) { o.append(s.data(), s.size()); return; }
  o.push_back('\x02');
  static const char* hx = "0123456789abcdef";
  for (unsigned char c : s) { o.push_back(hx[c >> 4]); o.push_back(hx[c & 15]); }
}
static inline void putInt(std::string& o, i64 v) {
  char b[24]; int n = 0;
  if (v == 0) { o.push_back('0'); return; }
  bool neg = v < 0; u64 u = neg ? -(u64)v : v;
  while (u) { b[n++] = '0' + u % 10; u /= 10; }
  if (neg) o.push_back('-');
  while (n) o.push_back(b[--n]);
}


// ---------------------------------------------------------------- parquet writer
// Minimal Parquet: PLAIN encoding, UNCOMPRESSED, one data page per column per row group.
struct Thrift {
  std::string b;
  int last = 0;
  std::vector<int> stk;
  void vi(u64 v) { while (v >= 0x80) { b.push_back((char)(v | 0x80)); v >>= 7; } b.push_back((char)v); }
  void zz(i64 v) { vi(((u64)v << 1) ^ (u64)(v >> 63)); }
  void field(int id, int type) {
    int d = id - last;
    if (d > 0 && d <= 15) b.push_back((char)((d << 4) | type));
    else { b.push_back((char)type); zz(id); }
    last = id;
  }
  void i32(int id, i64 v) { field(id, 5); zz(v); }
  void i64f(int id, i64 v) { field(id, 6); zz(v); }
  void str(int id, const char* s, size_t n) { field(id, 8); vi(n); b.append(s, n); }
  void str(int id, const std::string& s) { str(id, s.data(), s.size()); }
  void lst(int id, int etype, size_t n) {
    field(id, 9);
    if (n < 15) b.push_back((char)((n << 4) | etype));
    else { b.push_back((char)(0xF0 | etype)); vi(n); }
  }
  void begin(int id) { field(id, 12); stk.push_back(last); last = 0; }
  void beginElem() { stk.push_back(last); last = 0; }  // struct inside a list
  void end() { b.push_back(0); last = stk.back(); stk.pop_back(); }
};

struct PqCol { const char* name; int ptype; bool optional; int conv; };  // ptype: 1=INT32 2=INT64 6=BYTE_ARRAY; conv: -1 none, 0 UTF8, 6 DATE
struct PqWriter {
  int fd = -1;
  std::string path;
  const PqCol* cols; int ncols;
  u64 off = 0;
  i64 totalRows = 0;
  struct CC { i64 off, size, nvals; };
  struct RG { std::vector<CC> cc; i64 rows, bytes; };
  std::vector<RG> rgs;
  std::string wb;
  void open(const std::string& p, const PqCol* c, int n) {
    path = p; cols = c; ncols = n;
    fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    wb = "PAR1";
    off = 4;
  }
  void flushBuf() {
    const char* p = wb.data(); size_t n = wb.size();
    while (n) { ssize_t w = ::write(fd, p, n); if (w <= 0) break; p += w; n -= w; }
    wb.clear();
  }
  // add one column chunk (a single data page). body holds levels (if optional) + values.
  void addChunk(RG& rg, i64 nvals, const std::string& body) {
    Thrift h;
    h.i32(1, 0);                 // DATA_PAGE
    h.i32(2, (i64)body.size());  // uncompressed
    h.i32(3, (i64)body.size());  // compressed
    h.begin(5);                  // DataPageHeader
    h.i32(1, nvals);
    h.i32(2, 0);                 // PLAIN
    h.i32(3, 3);                 // def levels RLE
    h.i32(4, 3);                 // rep levels RLE
    h.end();
    h.b.push_back(0);            // end PageHeader
    CC cc; cc.off = off; cc.nvals = nvals; cc.size = (i64)(h.b.size() + body.size());
    wb.append(h.b); wb.append(body);
    off += cc.size;
    rg.cc.push_back(cc);
    rg.bytes += cc.size;
    if (wb.size() > (4u << 20)) flushBuf();
  }
  RG newRG(i64 rows) { RG r; r.rows = rows; r.bytes = 0; totalRows += rows; return r; }
  void endRG(RG& r) { rgs.push_back(std::move(r)); }
  void finish() {
    Thrift m;
    m.i32(1, 1);
    m.lst(2, 12, ncols + 1);
    m.beginElem(); m.str(4, "schema", 6); m.i32(5, ncols); m.end();
    for (int i = 0; i < ncols; i++) {
      m.beginElem();
      m.i32(1, cols[i].ptype);
      m.i32(3, cols[i].optional ? 1 : 0);
      m.str(4, cols[i].name, strlen(cols[i].name));
      if (cols[i].conv >= 0) m.i32(6, cols[i].conv);
      if (cols[i].conv == 0) { m.begin(10); m.begin(1); m.end(); m.end(); }
      else if (cols[i].conv == 6) { m.begin(10); m.begin(6); m.end(); m.end(); }
      m.end();
    }
    m.i64f(3, totalRows);
    m.lst(4, 12, rgs.size());
    for (auto& rg : rgs) {
      m.beginElem();
      m.lst(1, 12, ncols);
      for (int i = 0; i < ncols; i++) {
        m.beginElem();
        m.i64f(2, rg.cc[i].off);
        m.begin(3);
        m.i32(1, cols[i].ptype);
        if (cols[i].optional) { m.lst(2, 5, 2); m.zz(0); m.zz(3); }
        else { m.lst(2, 5, 1); m.zz(0); }
        m.lst(3, 8, 1); m.vi(strlen(cols[i].name)); m.b.append(cols[i].name);
        m.i32(4, 0);
        m.i64f(5, rg.cc[i].nvals);
        m.i64f(6, rg.cc[i].size);
        m.i64f(7, rg.cc[i].size);
        m.i64f(9, rg.cc[i].off);
        m.end();
        m.end();
      }
      m.i64f(2, rg.bytes);
      m.i64f(3, rg.rows);
      m.end();
    }
    m.str(6, "pipe", 4);
    m.b.push_back(0);
    wb.append(m.b);
    u32 ml = (u32)m.b.size();
    wb.append((const char*)&ml, 4);
    wb.append("PAR1");
    flushBuf();
    ::close(fd);
  }
};
static inline void put32(std::string& s, u32 v) { s.append((const char*)&v, 4); }
static inline void put64(std::string& s, u64 v) { s.append((const char*)&v, 8); }
static const PqCol UD_COLS[6] = {{"user_id", 6, false, 0}, {"day", 1, false, 6}, {"sessions", 1, false, -1},
                                 {"events", 1, false, -1}, {"revenue_cents", 2, false, -1}, {"top_category", 6, true, 0}};
static const PqCol CD_COLS[7] = {{"campaign_id", 6, false, 0}, {"channel", 6, false, 0}, {"day", 1, false, 6},
                                 {"sessions", 1, false, -1}, {"purchasing_sessions", 1, false, -1},
                                 {"revenue_cents", 2, false, -1}, {"assisted_sessions", 1, false, -1}};
static const int RGROWS = 1 << 19;
static inline void putBA(std::string& s, const char* p, size_t n) { put32(s, (u32)n); s.append(p, n); }

int main() {
  auto env = [](const char* k, const char* d) { const char* v = getenv(k); return std::string(v ? v : d); };
  DATA = env("DATA_DIR", "/data"); OUT = env("OUTPUT_DIR", "/output"); SCR = env("SCRATCH_DIR", "/scratch");
  PROF = getenv("PROFILE") != nullptr; T0 = nowms();
  NT = atoi(env("NCPU", "8").c_str()); if (NT < 1) NT = 1;
  mkdir(OUT.c_str(), 0755);
  mkdir((OUT + "/user_daily").c_str(), 0755);
  mkdir((OUT + "/campaign_daily").c_str(), 0755);
  // manifest
  {
    FILE* f = fopen((DATA + "/manifest.json").c_str(), "r");
    std::string s; char buf[4096]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, r);
    fclose(f);
    std::string a = jget(s, "window_start"), b = jget(s, "window_end");
    WS = parse_ts(sv(a)); WE = parse_ts(sv(b));
  }
  auto floordiv = [](i64 a, i64 b) { i64 q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return q; };
  DAY0 = floordiv(WS, DAY_MS);
  NDAYS = (int)(floordiv(WE - 1, DAY_MS) - DAY0 + 1);

  // ---- side thread 1: products + campaigns (small tables).
  u32 ORGANIC = 0; size_t NCAMP = 0;
  std::thread sideP([&]() {
    std::vector<std::pair<std::string, std::string>> prods;
    {
      MFile f = mapfile(DATA + "/products.csv");
      Task t{0, 0, f.n};
      if (f.n) for_lines(f, t, [&](const char* p, size_t n, bool sf) {
        sv fld[2];
        if (!split<2>(p, n, fld, sf)) return;
        if (fld[0].empty()) return;
        prods.push_back({std::string(fld[0]), std::string(fld[1])});
      });
    }
    {
      std::vector<std::string> cs;
      for (auto& p : prods) cs.push_back(p.second);
      cs.push_back("unknown");
      std::sort(cs.begin(), cs.end());
      cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
      catNames = cs;
      for (size_t i = 0; i < cs.size(); i++) if (cs[i] == "unknown") UNKCAT = i;
      prodHT.init(prods.size() + 1);
      for (auto& p : prods) {
        u16 c = std::lower_bound(cs.begin(), cs.end(), p.second) - cs.begin();
        bool ins; u32 id = prodCat.size();
        prodHT.getput(hashb(p.first.data(), p.first.size()), id, ins);
        if (ins) prodCat.push_back(c);
      }
    }
    {
      MFile f = mapfile(DATA + "/campaigns.csv");
      Task t{0, 0, f.n};
      std::vector<std::pair<std::string, std::string>> cp;
      if (f.n) for_lines(f, t, [&](const char* p, size_t n, bool sf) {
        sv fld[2];
        if (!split<2>(p, n, fld, sf)) return;
        if (fld[0].empty()) return;
        cp.push_back({std::string(fld[0]), std::string(fld[1])});
      });
      campHT.init(cp.size() + 1);
      for (auto& c : cp) {
        bool ins; u32 id = campId.size();
        campHT.getput(hashb(c.first.data(), c.first.size()), id, ins);
        if (ins) { campId.push_back(c.first); campChan.push_back(c.second); }
      }
      campId.push_back("organic"); campChan.push_back("organic");
    }
    ORGANIC = campId.size() - 1;
    NCAMP = campId.size();
  });

  // ---- side thread 2: links -> person lookup table
  std::thread sideL([&]() {
    std::vector<std::string> lnames;
    auto lfiles = readAll(DATA + "/links", lnames);
    std::vector<Row2> rows;
    {
      auto tasks = mktasks(lfiles, 8 << 20);
      for (auto& t : tasks)
        for_lines(lfiles[t.file], t, [&](const char* p, size_t n, bool sf) {
          sv f[2];
          if (!split<2>(p, n, f, sf)) return;
          if (f[0].empty() || f[1].empty()) return;
          rows.push_back({f[0], f[1]});
        });
    }
    size_t total = rows.size();
    linkHT.init(total * 2 + 1);
    std::vector<u32> parent;
    std::vector<sv> minv;
    std::vector<u64> nodeHash;
    parent.reserve(total * 2); minv.reserve(total * 2); nodeHash.reserve(total * 2);
    auto node = [&](sv s) -> u32 {
      bool ins; u32 id = parent.size();
      u64 hv = hashb(s.data(), s.size());
      u32 r = linkHT.getput(hv, id, ins);
      if (ins) { parent.push_back(id); minv.push_back(s); nodeHash.push_back(hv); }
      return r;
    };
    auto find = [&](u32 x) {
      while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };
    for (auto& r : rows) {
      u32 a = find(node(r.a)), b = find(node(r.b));
      if (a == b) continue;
      if (minv[b].compare(minv[a]) < 0) std::swap(a, b);  // keep a as smaller-named
      parent[b] = a;  // minv[a] stays min
    }
    size_t N = parent.size();
    std::vector<u64> rootKey(N);
    for (size_t i = 0; i < N; i++) {
      u32 r = find(i);
      if (r == i) rootKey[i] = hashb(minv[i].data(), minv[i].size());
    }
    if (N) {
      size_t cap = 16; while (cap < N * 2) cap <<= 1;
      lTab.assign(cap, LSlot{0, 0, 0, 0});
      lMask = cap - 1;
      for (size_t i = 0; i < N; i++) {
        u32 r = find(i);
        u64 hh = nodeHash[i] ? nodeHash[i] : 1;
        u64 k = hh & lMask;
        while (lTab[k].h != 0) k = (k + 1) & lMask;
        lTab[k] = LSlot{hh, rootKey[r], mkNm(minv[r].data(), minv[r].size()), 0};
      }
    }
  });
  TICK("setup");

  // ---- parse events and clicks (needs only the window; users/products/campaigns resolved later)
  std::vector<std::string> enames, cnames;
  auto efiles = readAll(DATA + "/events", enames);
  auto cfiles = readAll(DATA + "/clicks", cnames);
  std::vector<std::vector<std::vector<ERaw>>> raw(NT, std::vector<std::vector<ERaw>>(NPART));
  std::vector<std::vector<CRaw>> rawC(NT);
  {
    auto etasks = mktasks(efiles, 8 << 20);
    auto ctasks = mktasks(cfiles, 8 << 20);
    std::atomic<size_t> nxt{0};
    size_t etotal = 0; for (auto& f : efiles) etotal += f.n;
    size_t ctotal = 0; for (auto& f : cfiles) ctotal += f.n;
    size_t estLines = etotal / 64;  // deliberate over-estimate; untouched capacity costs only address space
    size_t nE = etasks.size(), nAll = etasks.size() + ctasks.size();
    run_threads(NT, [&](int tid) {
      auto& parts = raw[tid];
      auto& cout_ = rawC[tid];
      { size_t per = estLines / NT / NPART + estLines / NT / NPART / 3 + 32; for (auto& v : parts) v.reserve(per); }
      cout_.reserve(ctotal / 48 / NT + 1024);
      size_t i;
      while ((i = nxt++) < nAll) {
        if (i < nE) {
          const Task& t = etasks[i];
          for_lines(efiles[t.file], t, [&](const char* p, size_t n, bool sf) {
            sv f[7];
            if (!split<7>(p, n, f, sf)) return;
            if (f[0].empty()) return;
            ERaw r;
            r.eid = hashb(f[0].data(), f[0].size());
            i64 ing = parse_ts(f[3]);
            r.ing = ing == BADTS ? 0xFFFFFFFFu : (ing < 0 ? 0u : (ing / 1000 > 0xFFFFFFFELL ? 0xFFFFFFFEu : (u32)(ing / 1000)));
            r.ts = 0; r.uh = 0; r.unm = 0; r.ph = 0; r.meta = 0;
            bool ok = !f[1].empty() && !f[5].empty();
            i64 ts = BADTS;
            if (ok) {
              ts = parse_ts(f[2]);
              ok = ts != BADTS && ts >= WS && ts < WE && (ing == BADTS ? false : ing - ts <= DAY_MS);
            }
            u32 type = 0, price = 0;
            if (ok) {
              sv ty = f[4];
              if (ty == "view") type = 0;
              else if (ty == "cart") type = 1;
              else if (ty == "purchase") type = 2;
              else ok = false;
            }
            if (ok && type == 2) {
              sv pr = f[6];
              if (pr.empty()) ok = false;
              else {
                size_t k = 0;
                while (k < pr.size() && pr[k] == '0') k++;
                if (pr.size() - k > 8) ok = false;
                else {
                  u64 v = 0;
                  for (size_t j = 0; j < pr.size(); j++) {
                    if (!isdig(pr[j])) { ok = false; break; }
                    if (j >= k) v = v * 10 + (pr[j] - '0');
                  }
                  if (ok && (v < 1 || v > 10000000)) ok = false;
                  price = (u32)v;
                }
              }
            }
            if (ok) {
              r.ts = ts;
              r.meta = price | (type << 24) | (1u << 26);
              if (type == 2) r.ph = hashb(f[5].data(), f[5].size());
              r.uh = hashb(f[1].data(), f[1].size());
              r.unm = mkNm(f[1].data(), f[1].size());
            }
            parts[partOf(r.eid)].push_back(r);
          });
        } else {
          const Task& t = ctasks[i - nE];
          for_lines(cfiles[t.file], t, [&](const char* p, size_t n, bool sf) {
            sv f[4];
            if (!split<4>(p, n, f, sf)) return;
            if (f[0].empty() || f[1].empty()) return;
            i64 ts = parse_ts(f[3]);
            if (ts == BADTS || ts < WS - WEEK_MS || ts >= WE) return;
            CRaw c;
            c.uh = hashb(f[1].data(), f[1].size());
            c.ts = ts;
            c.cid = mkNm(f[0].data(), f[0].size());
            c.ch = hashb(f[2].data(), f[2].size());
            c.unm = mkNm(f[1].data(), f[1].size());
            cout_.push_back(c);
          });
        }
      }
    });
  }
  TICK("parse");
  sideP.join();
  sideL.join();
  TICK("side-join");

  // ---- events: dedup + resolve persons/categories -> person buckets; clicks: resolve -> person buckets
  std::vector<std::vector<std::vector<ERec>>> ev(NT, std::vector<std::vector<ERec>>(NBUCK));
  std::vector<std::vector<std::vector<CRec>>> ck(NT, std::vector<std::vector<CRec>>(NBUCK));
  {
    struct CChunk { int t; size_t s, e; };
    std::vector<CChunk> cchunks;
    for (int t = 0; t < NT; t++)
      for (size_t s = 0; s < rawC[t].size(); s += 32768) cchunks.push_back({t, s, std::min(rawC[t].size(), s + 32768)});
    std::atomic<size_t> nxt{0};
    size_t total = (size_t)NPART + cchunks.size();
    size_t nRaw = 0, nCl = 0;
    for (int t = 0; t < NT; t++) { for (int p = 0; p < NPART; p++) nRaw += raw[t][p].size(); nCl += rawC[t].size(); }
    run_threads(NT, [&](int tid) {
      auto& out = ev[tid];
      auto& cout_ = ck[tid];
      { size_t per = nRaw / NT / NBUCK; per += per / 3 + 8; for (auto& v : out) v.reserve(per); }
      { size_t per = nCl / NT / NBUCK; per += per / 3 + 4; for (auto& v : cout_) v.reserve(per); }
      std::vector<u32> tab;
      std::vector<const ERaw*> recs, surv;
      size_t part;
      while ((part = nxt++) < total) {
        if (part >= (size_t)NPART) {
          const CChunk& cc = cchunks[part - NPART];
          const CRaw* base = rawC[cc.t].data();
          size_t cnt = cc.e - cc.s;
          for (size_t k = 0; k < cnt; k++) {
            if (k + 8 < cnt) prefetchUser(base[cc.s + k + 8].uh);
            const CRaw& c = base[cc.s + k];
            u32 camp = campHT.get(c.ch);
            if (camp == 0xFFFFFFFFu) continue;
            CRec r; u64 nm;
            sv u = getNm(c.unm);
            resolveUserH(u, c.uh, r.pkey, nm);
            sv cid = getNm(c.cid);
            r.ts = c.ts; r.cid = cid.data(); r.cidlen = (u32)cid.size(); r.camp = camp;
            cout_[bucketOf(r.pkey)].push_back(r);
          }
          continue;
        }
        size_t tot = 0;
        for (int t = 0; t < NT; t++) tot += raw[t][part].size();
        if (!tot) continue;
        recs.clear(); recs.reserve(tot);
        for (int t = 0; t < NT; t++) for (auto& r : raw[t][part]) recs.push_back(&r);
        size_t cap = 16; while (cap < tot * 2) cap <<= 1;
        tab.assign(cap, 0xFFFFFFFFu);
        u64 m = cap - 1;
        for (size_t i = 0; i < tot; i++) {
          const ERaw* r = recs[i];
          u64 h = (r->eid * 0x9E3779B97F4A7C15ULL) >> 20;
          u64 k = h & m;
          while (true) {
            u32 j = tab[k];
            if (j == 0xFFFFFFFFu) { tab[k] = i; break; }
            const ERaw* o = recs[j];
            if (o->eid == r->eid) { if (r->ing < o->ing) tab[k] = i; break; }
            k = (k + 1) & m;
          }
        }
        surv.clear();
        for (size_t k = 0; k < cap; k++) {
          u32 j = tab[k];
          if (j == 0xFFFFFFFFu) continue;
          const ERaw* r = recs[j];
          if (!((r->meta >> 26) & 1)) continue;
          surv.push_back(r);
        }
        size_t ns = surv.size();
        for (size_t k = 0; k < ns; k++) {
          if (k + 8 < ns) prefetchUser(surv[k + 8]->uh);
          const ERaw* r = surv[k];
          ERec e;
          sv u = getNm(r->unm);
          resolveUserH(u, r->uh, e.pkey, e.nm);
          e.ts = r->ts; e.meta = r->meta & 0x3FFFFFF; e.cat = 0; e.pad = 0;
          if (((r->meta >> 24) & 3) == 2) {
            u32 pc = prodHT.get(r->ph);
            e.cat = pc == 0xFFFFFFFFu ? UNKCAT : prodCat[pc];
          }
          out[bucketOf(e.pkey)].push_back(e);
        }
        for (int t = 0; t < NT; t++) { std::vector<ERaw>().swap(raw[t][part]); }
      }
    });
  }
  TICK("dedup+clicks");
  // ---- per-bucket processing
  struct TA {
    std::vector<i64> sess, psess, rev, asst;
    // buffered user_daily rows
    std::vector<sv> uName; std::vector<i32_t> uDay, uSess, uEv; std::vector<i64> uRev; std::vector<u16> uCat;
    PqWriter pw; bool opened = false;
  };
  std::vector<TA> ta(NT);
  size_t CELLS = NCAMP * (size_t)NDAYS;
  {
    std::atomic<size_t> nxt{0};
    run_threads(NT, [&](int tid) {
      TA& A = ta[tid];
      A.sess.assign(CELLS, 0); A.psess.assign(CELLS, 0); A.rev.assign(CELLS, 0); A.asst.assign(CELLS, 0);
      auto flushUD = [&]() {
        size_t n = A.uName.size();
        if (!n) return;
        if (!A.opened) { A.opened = true; A.pw.open(OUT + "/user_daily/part-" + std::to_string(tid) + ".parquet", UD_COLS, 6); }
        PqWriter::RG rg = A.pw.newRG(n);
        std::string body;
        body.reserve(n * 24);
        for (size_t i = 0; i < n; i++) putBA(body, A.uName[i].data(), A.uName[i].size());
        A.pw.addChunk(rg, n, body);
        body.clear(); body.append((const char*)A.uDay.data(), n * 4);
        A.pw.addChunk(rg, n, body);
        body.clear(); body.append((const char*)A.uSess.data(), n * 4);
        A.pw.addChunk(rg, n, body);
        body.clear(); body.append((const char*)A.uEv.data(), n * 4);
        A.pw.addChunk(rg, n, body);
        body.clear(); body.append((const char*)A.uRev.data(), n * 8);
        A.pw.addChunk(rg, n, body);
        // top_category: optional
        body.clear();
        size_t groups = (n + 7) / 8;
        std::string lv;
        {
          Thrift t; t.vi((groups << 1) | 1);
          lv = t.b;
          size_t base = lv.size();
          lv.resize(base + groups, 0);
          for (size_t i = 0; i < n; i++) if (A.uCat[i] != 0xFFFF) lv[base + (i >> 3)] |= (char)(1 << (i & 7));
        }
        put32(body, (u32)lv.size());
        body.append(lv);
        for (size_t i = 0; i < n; i++) if (A.uCat[i] != 0xFFFF) { const std::string& cn = catNames[A.uCat[i]]; putBA(body, cn.data(), cn.size()); }
        A.pw.addChunk(rg, n, body);
        A.pw.endRG(rg);
        A.uName.clear(); A.uDay.clear(); A.uSess.clear(); A.uEv.clear(); A.uRev.clear(); A.uCat.clear();
      };
      std::vector<ERec> E, E2;
      std::vector<CRec> C, C2;
      std::vector<u64> hKey; std::vector<u32> hGid, gOf, gCnt, gB, gPos, cOf, cB, cPos;
      std::vector<i64> sStart, sEv, sRev, sCredCamp;
      std::vector<u32> sCamp;
      std::vector<std::pair<u32, i64>> cc;
      std::vector<u32> dayfirst;
      std::vector<i64> catSum(catNames.size(), 0);
      std::vector<u16> touched;
      std::vector<u32> gf;
      size_t b;
      while ((b = nxt++) < (size_t)NBUCK) {
        E.clear(); C.clear();
        for (int t = 0; t < NT; t++) {
          E.insert(E.end(), ev[t][b].begin(), ev[t][b].end());
          C.insert(C.end(), ck[t][b].begin(), ck[t][b].end());
          std::vector<ERec>().swap(ev[t][b]);
          std::vector<CRec>().swap(ck[t][b]);
        }
        if (E.empty()) continue;
        // group events by person with a small hash map, then order each group by ts
        size_t ne = E.size();
        size_t hcap = 16; while (hcap < ne * 2) hcap <<= 1;
        hKey.assign(hcap, 0); hGid.assign(hcap, 0xFFFFFFFFu);
        u64 hm = hcap - 1;
        gOf.resize(ne); gCnt.clear();
        for (size_t k = 0; k < ne; k++) {
          u64 pk = E[k].pkey;
          u64 x = (pk * 0x9E3779B97F4A7C15ULL) >> 24;
          u64 q = x & hm;
          while (true) {
            if (hGid[q] == 0xFFFFFFFFu) { hGid[q] = (u32)gCnt.size(); hKey[q] = pk; gCnt.push_back(0); break; }
            if (hKey[q] == pk) break;
            q = (q + 1) & hm;
          }
          u32 g = hGid[q];
          gOf[k] = g; gCnt[g]++;
        }
        size_t NG = gCnt.size();
        gB.assign(NG + 1, 0);
        for (size_t g = 0; g < NG; g++) gB[g + 1] = gB[g] + gCnt[g];
        gPos.assign(gB.begin(), gB.end() - 1);
        E2.resize(ne);
        for (size_t k = 0; k < ne; k++) E2[gPos[gOf[k]]++] = E[k];
        for (size_t g = 0; g < NG; g++) {
          ERec* p = E2.data() + gB[g]; size_t n = gB[g + 1] - gB[g];
          if (n <= 24) {
            for (size_t x = 1; x < n; x++) {
              ERec v = p[x]; size_t y = x;
              while (y > 0 && p[y - 1].ts > v.ts) { p[y] = p[y - 1]; y--; }
              p[y] = v;
            }
          } else std::sort(p, p + n, [](const ERec& a, const ERec& b) { return a.ts < b.ts; });
        }
        // clicks to groups (clicks of people without events are irrelevant)
        size_t nc = C.size();
        cB.assign(NG + 1, 0);
        cOf.resize(nc);
        for (size_t k = 0; k < nc; k++) {
          u64 pk = C[k].pkey;
          u64 x = (pk * 0x9E3779B97F4A7C15ULL) >> 24;
          u64 q = x & hm;
          u32 g = 0xFFFFFFFFu;
          while (hGid[q] != 0xFFFFFFFFu) {
            if (hKey[q] == pk) { g = hGid[q]; break; }
            q = (q + 1) & hm;
          }
          cOf[k] = g;
          if (g != 0xFFFFFFFFu) cB[g + 1]++;
        }
        for (size_t g = 0; g < NG; g++) cB[g + 1] += cB[g];
        cPos.assign(cB.begin(), cB.end() - 1);
        C2.resize(cB[NG]);
        for (size_t k = 0; k < nc; k++) if (cOf[k] != 0xFFFFFFFFu) C2[cPos[cOf[k]]++] = C[k];
        for (size_t g = 0; g < NG; g++) {
          CRec* p = C2.data() + cB[g]; size_t n = cB[g + 1] - cB[g];
          if (n > 1) std::sort(p, p + n, [](const CRec& a, const CRec& b) {
            if (a.ts != b.ts) return a.ts < b.ts;
            int c = memcmp(a.cid, b.cid, std::min(a.cidlen, b.cidlen));
            if (c) return c < 0;
            return a.cidlen < b.cidlen;
          });
        }
        for (size_t g = 0; g < NG; g++) {
          size_t i = gB[g], j = gB[g + 1];
          size_t cs = cB[g], ce = cB[g + 1];
          // dedup clicks (sorted by ts,cid)
          size_t cn = 0;
          for (size_t k = cs; k < ce; k++) {
            if (cn > 0) {
              const CRec& pr = C2[cs + cn - 1];
              if (pr.ts == C2[k].ts && pr.cidlen == C2[k].cidlen && memcmp(pr.cid, C2[k].cid, pr.cidlen) == 0) continue;
            }
            C2[cs + cn] = C2[k]; cn++;
          }
          CRec* cl = C2.data() + cs;
          // group-first index
          gf.resize(cn);
          for (size_t k = 0; k < cn; k++) gf[k] = (k > 0 && cl[k].ts == cl[k - 1].ts) ? gf[k - 1] : (u32)k;
          // sessions
          // compute session boundaries (indices into E)
          static thread_local std::vector<u32> sBeg;
          sBeg.clear();
          for (size_t k = i; k < j; k++) {
            if (k == i || E2[k].ts - E2[k - 1].ts > 1800000) sBeg.push_back(k);
          }
          size_t S = sBeg.size();
          sBeg.push_back(j);
          // credit
          size_t cp = 0;  // number of clicks with ts <= start
          sv name = getNm(E2[i].nm);
          // per-day aggregation for user_daily
          i64 curDay = -1, dSess = 0, dEv = 0, dRev = 0;
          auto flush = [&]() {
            if (curDay < 0) return;
            u16 bestc = 0xFFFF;
            if (!touched.empty()) {
              u16 best = touched[0];
              for (u16 c : touched) if (catSum[c] > catSum[best] || (catSum[c] == catSum[best] && c < best)) best = c;
              bestc = best;
              for (u16 c : touched) catSum[c] = 0;
              touched.clear();
            }
            A.uName.push_back(name); A.uDay.push_back((i32_t)(DAY0 + curDay)); A.uSess.push_back((i32_t)dSess);
            A.uEv.push_back((i32_t)dEv); A.uRev.push_back(dRev); A.uCat.push_back(bestc);
            if (A.uName.size() >= (size_t)RGROWS) flushUD();
          };
          sStart.resize(S);
          for (size_t s = 0; s < S; s++) {
            size_t b0 = sBeg[s], b1 = sBeg[s + 1];
            i64 start = E2[b0].ts;
            sStart[s] = start;
            i64 day = floordiv(start, DAY_MS) - DAY0;
            if (day != curDay) { flush(); curDay = day; dSess = dEv = dRev = 0; }
            i64 rv = 0;
            for (size_t k = b0; k < b1; k++) {
              if (((E2[k].meta >> 24) & 3) == 2) {
                u32 pr = E2[k].meta & 0xFFFFFF;
                rv += pr;
                u16 c = E2[k].cat;
                if (catSum[c] == 0) touched.push_back(c);
                catSum[c] += pr;
              }
            }
            dSess++; dEv += b1 - b0; dRev += rv;
            // credit
            while (cp < cn && cl[cp].ts <= start) cp++;
            u32 camp = ORGANIC;
            if (cp > 0 && start - cl[cp - 1].ts <= WEEK_MS) camp = cl[gf[cp - 1]].camp;
            size_t cell = (size_t)camp * NDAYS + day;
            A.sess[cell]++;
            if (rv > 0) A.psess[cell]++;
            A.rev[cell] += rv;
          }
          flush();
          // assists
          if (cn > 0) {
            cc.clear();
            for (size_t k = 0; k < cn; k++) cc.push_back({cl[k].camp, cl[k].ts});
            if (cn > 1) std::sort(cc.begin(), cc.end());
            bool dfReady = false;
            auto addRange = [&](i64 a, i64 bnd) {
              size_t lo = std::lower_bound(sStart.begin(), sStart.begin() + S, a) - sStart.begin();
              size_t hi = std::upper_bound(sStart.begin(), sStart.begin() + S, bnd) - sStart.begin();
              return std::make_pair(lo, hi);
            };
            size_t x = 0;
            while (x < cc.size()) {
              u32 camp = cc[x].first;
              i64 a = cc[x].second, bnd = a + WEEK_MS;
              size_t y = x + 1;
              auto emit = [&](i64 A0, i64 B0) {
                auto [lo, hi] = addRange(A0, B0);
                if (lo >= hi) return;
                if (hi - lo <= 12) {
                  for (size_t s = lo; s < hi; s++) A.asst[(size_t)camp * NDAYS + (floordiv(sStart[s], DAY_MS) - DAY0)]++;
                } else {
                  if (!dfReady) {
                    dayfirst.assign(NDAYS + 1, (u32)S);
                    for (int d = NDAYS; d >= 0; d--) dayfirst[d] = (u32)S;
                    // dayfirst[d] = first session idx with day >= d
                    size_t s = 0;
                    for (int d = 0; d <= NDAYS; d++) {
                      while (s < S && floordiv(sStart[s], DAY_MS) - DAY0 < d) s++;
                      dayfirst[d] = (u32)s;
                    }
                    dfReady = true;
                  }
                  i64 d0 = floordiv(sStart[lo], DAY_MS) - DAY0, d1 = floordiv(sStart[hi - 1], DAY_MS) - DAY0;
                  for (i64 d = d0; d <= d1; d++) {
                    i64 l2 = std::max<i64>(lo, dayfirst[d]), h2 = std::min<i64>(hi, dayfirst[d + 1]);
                    if (h2 > l2) A.asst[(size_t)camp * NDAYS + d] += h2 - l2;
                  }
                }
              };
              for (; y < cc.size() && cc[y].first == camp; y++) {
                if (cc[y].second <= bnd) bnd = cc[y].second + WEEK_MS;
                else { emit(a, bnd); a = cc[y].second; bnd = a + WEEK_MS; }
              }
              emit(a, bnd);
              x = y;
            }
          }
        }
      }
      flushUD();
      if (A.opened) A.pw.finish();
    });
  }
  TICK("buckets");
  // ---- campaign output
  {
    std::vector<u32> rc; std::vector<i32_t> rd; std::vector<i64> rs, rp, rr, ra;
    for (size_t c = 0; c < NCAMP; c++)
      for (int d = 0; d < NDAYS; d++) {
        size_t cell = c * NDAYS + d;
        i64 sn = 0, ps = 0, rv = 0, as = 0;
        for (int t = 0; t < NT; t++) { sn += ta[t].sess[cell]; ps += ta[t].psess[cell]; rv += ta[t].rev[cell]; as += ta[t].asst[cell]; }
        if (!sn && !as) continue;
        rc.push_back((u32)c); rd.push_back((i32_t)(DAY0 + d)); rs.push_back(sn); rp.push_back(ps); rr.push_back(rv); ra.push_back(as);
      }
    PqWriter pw; pw.open(OUT + "/campaign_daily/part-0.parquet", CD_COLS, 7);
    size_t n = rc.size();
    if (n) {
      PqWriter::RG rg = pw.newRG(n);
      std::string body;
      for (size_t i = 0; i < n; i++) putBA(body, campId[rc[i]].data(), campId[rc[i]].size());
      pw.addChunk(rg, n, body); body.clear();
      for (size_t i = 0; i < n; i++) putBA(body, campChan[rc[i]].data(), campChan[rc[i]].size());
      pw.addChunk(rg, n, body); body.clear();
      for (size_t i = 0; i < n; i++) put32(body, (u32)rd[i]);
      pw.addChunk(rg, n, body); body.clear();
      for (size_t i = 0; i < n; i++) put32(body, (u32)rs[i]);
      pw.addChunk(rg, n, body); body.clear();
      for (size_t i = 0; i < n; i++) put32(body, (u32)rp[i]);
      pw.addChunk(rg, n, body); body.clear();
      for (size_t i = 0; i < n; i++) put64(body, (u64)rr[i]);
      pw.addChunk(rg, n, body); body.clear();
      for (size_t i = 0; i < n; i++) put32(body, (u32)ra[i]);
      pw.addChunk(rg, n, body);
      pw.endRG(rg);
    }
    pw.finish();
  }
  {
    bool any = false;
    for (int t = 0; t < NT; t++) any |= ta[t].opened;
    if (!any) { PqWriter pw; pw.open(OUT + "/user_daily/part-0.parquet", UD_COLS, 6); pw.finish(); }
  }
  TICK("output");
  _exit(0);
}
