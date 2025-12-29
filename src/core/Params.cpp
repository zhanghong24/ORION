#include "core/Params.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <stdexcept>

namespace orion {
namespace core{

static inline std::string trim(std::string s) {
  auto issp = [](unsigned char c){ return std::isspace(c); };
  while(!s.empty() && issp(s.front())) s.erase(s.begin());
  while(!s.empty() && issp(s.back()))  s.pop_back();
  return s;
}

static inline std::string strip_comment(const std::string& s) {
  // Fortran 常见注释：!
  auto pos = s.find('!');
  if (pos == std::string::npos) return s;
  return s.substr(0, pos);
}

static inline std::string lower(std::string s) {
  for (auto& c: s) c = (char)std::tolower((unsigned char)c);
  return s;
}

static inline bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// 粗略 token 化：把逗号当分隔符；保留 = 以便解析 key/value
static std::vector<std::string> split_tokens(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool in_quote = false;
  char quote_ch = 0;

  auto flush = [&]{
    auto t = trim(cur);
    if (!t.empty()) out.push_back(t);
    cur.clear();
  };

  for (size_t i=0;i<line.size();++i) {
    char c = line[i];
    if (!in_quote && (c=='\'' || c=='"')) {
      in_quote = true; quote_ch = c;
      cur.push_back(c);
      continue;
    }
    if (in_quote) {
      cur.push_back(c);
      if (c == quote_ch) in_quote = false;
      continue;
    }
    if (c==',' ) { flush(); continue; }
    cur.push_back(c);
  }
  flush();
  return out;
}

static inline std::string unquote(std::string s) {
  s = trim(s);
  if (s.size()>=2) {
    if ((s.front()=='"' && s.back()=='"') || (s.front()=='\'' && s.back()=='\'')) {
      return s.substr(1, s.size()-2);
    }
  }
  return s;
}

template <class T>
static T parse_number(const std::string& s);

template <>
int parse_number<int>(const std::string& s) {
  return std::stoi(trim(s));
}

template <>
double parse_number<double>(const std::string& s) {
  std::string t = trim(s);
  // Fortran double exponent: 8.819D-4 / 1.0d+03  ->  8.819E-4 / 1.0e+03
  for (auto& c : t) {
    if (c == 'D' || c == 'd') c = 'E';
  }
  return std::stod(t);
}

// 一个小工具：把 "key = value" 切开
static bool split_kv(const std::string& token, std::string& key, std::string& val) {
  auto pos = token.find('=');
  if (pos == std::string::npos) return false;
  key = trim(token.substr(0,pos));
  val = trim(token.substr(pos+1));
  return true;
}

void Params::load_namelist_file(const std::string& path)
{
  std::ifstream fin(path);
  if (!fin) throw std::runtime_error("Failed to open param file: " + path);

  std::string line;
  std::string current_group;
  std::ostringstream block;

  auto commit_block = [&](const std::string& group, const std::string& content){
    if (group.empty()) return;

    // 将整个 block 内容按 token 解析
    std::istringstream iss(content);
    std::string l;
    while (std::getline(iss, l)) {
      l = strip_comment(l);
      l = trim(l);
      if (l.empty()) continue;

      // 一行可能有多个项，用逗号切
      auto toks = split_tokens(l);
      for (auto& t : toks) {
        std::string k,v;
        if (!split_kv(t, k, v)) continue;
        k = lower(trim(k));
        v = trim(v);

        // 下面：按 group 分发赋值（等价于 Fortran namelist 写入 global vars）
        const auto g = lower(trim(group));

        try {
          if (g=="inflow") {
            if (k=="moo") inflow.moo = parse_number<double>(v);
            else if (k=="reynolds") inflow.reynolds = parse_number<double>(v);
            else if (k=="attack") inflow.attack = parse_number<double>(v);
            else if (k=="sideslip") inflow.sideslip = parse_number<double>(v);
            else if (k=="tref") inflow.tref = parse_number<double>(v);
            else if (k=="twall") inflow.twall = parse_number<double>(v);
            else if (k=="pref") inflow.pref = parse_number<double>(v);
            else if (k=="rref") inflow.rref = parse_number<double>(v);
            else if (k=="vref") inflow.vref = parse_number<double>(v);
            else if (k=="ndim") inflow.ndim = parse_number<int>(v);
            else if (k=="height") inflow.height = parse_number<double>(v);
            else std::cerr << "[Params] warn: unknown inflow key: " << k << "\n";
          }
          else if (g=="force") {
            if (k=="nwholefield") force.nwholefield = parse_number<int>(v);
            else if (k=="sref") force.sref = parse_number<double>(v);
            else if (k=="lfref") force.lfref = parse_number<double>(v);
            else if (k=="lref") force.lref = parse_number<double>(v);
            else if (k=="xref") force.xref = parse_number<double>(v);
            else if (k=="yref") force.yref = parse_number<double>(v);
            else if (k=="zref") force.zref = parse_number<double>(v);
            else std::cerr << "[Params] warn: unknown force key: " << k << "\n";
          }
          else if (g=="filename") {
            if (k=="flowname") filename.flowname = unquote(v);
            else if (k=="gridname") filename.gridname = unquote(v);
            else if (k=="bcname") filename.bcname = unquote(v);
            else if (k=="forcename") filename.forcename = unquote(v);
            else if (k=="errhis") filename.errhis = unquote(v);
            else std::cerr << "[Params] warn: unknown filename key: " << k << "\n";
          }
          else if (g=="control") {
            if (k=="nstart") control.nstart = parse_number<int>(v);
            else if (k=="nmethod") control.nmethod = parse_number<int>(v);
            else if (k=="nbgmax") control.nbgmax = parse_number<int>(v);
            else if (k=="ndisk") control.ndisk = parse_number<int>(v);
            else if (k=="newton") control.newton = parse_number<int>(v);
            else if (k=="nomax") control.nomax = parse_number<int>(v);
            else if (k=="nforce") control.nforce = parse_number<int>(v);
            else if (k=="nwerror") control.nwerror = parse_number<int>(v);
            else if (k=="method") control.method = parse_number<int>(v);
            else if (k=="nplot") control.nplot = parse_number<int>(v);
            else std::cerr << "[Params] warn: unknown control key: " << k << "\n";
          }
          else if (g=="step") {
            if (k=="ntmst") step.ntmst = parse_number<int>(v);
            else if (k=="cfl") step.cfl = parse_number<double>(v);
            else if (k=="timedt") step.timedt = parse_number<double>(v);
            else if (k=="timedt_rate") step.timedt_rate = parse_number<double>(v);
            else if (k=="dtdts") step.dtdts = parse_number<double>(v);
            else if (k=="nsubstmx") step.nsubstmx = parse_number<int>(v);
            else if (k=="tolsub") step.tolsub = parse_number<double>(v);
            else std::cerr << "[Params] warn: unknown step key: " << k << "\n";
          }
          else if (g=="flowtype") {
            if (k=="nvis") flowtype.nvis = parse_number<int>(v);
            else if (k=="nchem") flowtype.nchem = parse_number<int>(v);
            else if (k=="ntmodel") flowtype.ntmodel = parse_number<int>(v);
            else std::cerr << "[Params] warn: unknown flowtype key: " << k << "\n";
          }
          else if (g=="technic") {
            if (k=="nlhs") technic.nlhs = parse_number<int>(v);
            else if (k=="nscheme") technic.nscheme = parse_number<int>(v);
            else if (k=="nlimiter") technic.nlimiter = parse_number<int>(v);
            else if (k=="efix") technic.efix = parse_number<int>(v);
            else if (k=="csrv") technic.csrv = parse_number<int>(v);
            else if (k=="nsmooth") technic.nsmooth = parse_number<int>(v);
            else if (k=="nflux") technic.nflux = parse_number<int>(v);
            else std::cerr << "[Params] warn: unknown technic key: " << k << "\n";
          }
          else if (g=="interplate") {
            if (k=="xk") interplate.xk = parse_number<double>(v);
            else if (k=="xb") interplate.xb = parse_number<double>(v);
            else if (k=="c_k2") interplate.c_k2 = parse_number<double>(v);
            else if (k=="c_k4") interplate.c_k4 = parse_number<double>(v);
            else std::cerr << "[Params] warn: unknown interplate key: " << k << "\n";
          }
          else if (g=="gcl_cic") {
            if (k=="gcl") gcl_cic.gcl = parse_number<int>(v);
            else if (k=="cic1") gcl_cic.cic1 = parse_number<double>(v);
            else if (k=="cic2") gcl_cic.cic2 = parse_number<double>(v);
            else if (k=="cic3") gcl_cic.cic3 = parse_number<double>(v);
            else if (k=="cic4") gcl_cic.cic4 = parse_number<double>(v);
            else if (k=="cic5") gcl_cic.cic5 = parse_number<double>(v);
            else std::cerr << "[Params] warn: unknown gcl_cic key: " << k << "\n";
          }
          else if (g=="connect_0") {
            if (k=="connect_point") connect_0.connect_point = parse_number<int>(v);
            else if (k=="conpointname") connect_0.conpointname = unquote(v);
            else if (k=="connect_order") connect_0.connect_order = parse_number<int>(v);
            else if (k=="dis_tao") connect_0.dis_tao = parse_number<double>(v);
            else std::cerr << "[Params] warn: unknown connect_0 key: " << k << "\n";
          }
          else {
            std::cerr << "[Params] warn: unknown group: " << group << " (key=" << k << ")\n";
          }
        } catch (const std::exception& e) {
          std::cerr << "[Params] error parsing " << group << "." << k << " = " << v
                    << " : " << e.what() << "\n";
          throw;
        }
      }
    }
  };

  while (std::getline(fin, line)) {
    line = strip_comment(line);
    line = trim(line);
    if (line.empty()) continue;

    // group begin: &name  or  $name
    if (!line.empty() && (line[0] == '&' || line[0] == '$')) {
    // commit previous
    commit_block(current_group, block.str());
    block.str(""); block.clear();

    current_group = trim(line.substr(1)); // skip '&' or '$'
    current_group = lower(current_group); // group name case-insensitive
    continue;
    }

    // group end: /
    if (trim(line) == "/") {
        commit_block(current_group, block.str());
        block.str(""); block.clear();
        current_group.clear();
        continue;
    }

    // inside group
    if (!current_group.empty()) {
      block << line << "\n";
    }
  }

  // commit tail (in case file doesn't end with '/')
  commit_block(current_group, block.str());

  postprocess();
}

void Params::print_summary() const
{
  std::cout << "---- ORION Params Summary ----\n";
  std::cout << "[inflow] moo=" << inflow.moo
            << " reynolds=" << inflow.reynolds
            << " attack=" << inflow.attack
            << " sideslip=" << inflow.sideslip
            << " ndim=" << inflow.ndim
            << " height=" << inflow.height << "\n";

  std::cout << "[filename] flowname=" << filename.flowname
            << " gridname=" << filename.gridname
            << " bcname=" << filename.bcname << "\n";

  std::cout << "[control] nstart=" << control.nstart
            << " nmethod=" << control.nmethod
            << " nplot=" << control.nplot << "\n";

  std::cout << "------------------------------\n";
}

} // namespae core
} // namespace orion
