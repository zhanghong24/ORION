#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace orion {
namespace core{

struct Params {

  struct Inflow {
    double moo=0, reynolds=0, attack=0, sideslip=0;
    double tref=0, twall=0, pref=0, rref=0, vref=0;
    int ndim=0;
    double height=0;

    // 气体物性参数 (原本可能散落在其他地方，建议统一放在这里或单独的 GasModel)
    double gama = 1.4;      // gamma
    double rjmk = 8.31434;    // 气体常数 R
    double prl = 0.72;      // 层流普朗特数
    double prt = 0.9;       // 湍流普朗特数
    
    // 组分相关 (Fortran 中的 ns, ws, cn_init)
    int ns = 1;
    std::vector<double> ws;      // 分子量
    std::vector<double> cn_init; // 初始浓度

    // --- [新增] init_inflow 计算出的衍生变量 (Derived) ---
    
    // 1. 方向余弦与无量纲量
    double uoo = 0.0, voo = 0.0, woo = 0.0;
    double roo = 1.0, too = 1.0; // 无量纲密度/温度 (通常归一化为1)
    double poo = 0.0, eoo = 0.0, hoo = 0.0; // 无量纲压力/能量/焓

    // 2. 守恒变量无穷远初值 (用于给流场赋初值)
    double q1oo=0, q2oo=0, q3oo=0, q4oo=0, q5oo=0;

    // 3. 参考量与系数
    double mref = 0.0;   // 混合气体平均分子量
    double mref1 = 0.0;  // 1/mref
    double visloo = 0.0; // 参考粘性系数
    double ccoo = 0.0;   // 音速
    double ccon = 0.0;   // Sutherland 常数项
    
    // 4. 无量纲计算系数
    double cq_lam = 0.0; // 热流系数 (层流)
    double cq_tur = 0.0; // 热流系数 (湍流)
    double visc = 0.0;   // 粘性相关
    double re = 0.0;     // 1.0 / reynolds
    
    // 5. 长度参考 (Fortran: lfref, lref)
    double lref = 1.0;
    double lfref = 1.0;

    // 辅助
    std::vector<double> ws1;
    std::vector<double> ms;
    std::vector<double> ms1;
  } inflow;

  struct Control {
    int nstart=0, nmethod=0, nbgmax=0, ndisk=0;
    int newton=0, nomax=0, nforce=0, nwerror=0;
    int method=0, nplot=0;
  } control;

  struct Step {
    int ntmst=0;
    double cfl=0, timedt=0, timedt_rate=0, dtdts=0;
    int nsubstmx=0;
    double tolsub=0;
    double phydtime=0.0;
  } step;

  struct Flowtype {
    int nvis=0, nchem=0, ntmodel=0;
  } flowtype;

  struct Technic {
    int nlhs=0, nscheme=0, nlimiter=0, efix=0, csrv=0, nsmooth=0, nflux=0;
  } technic;

  struct Interplate {
    double xk=0, xb=0, c_k2=0, c_k4=0;
  } interplate;

  struct Filename {
    std::string flowname, gridname, bcname, forcename, errhis;
  } filename;

  struct Force {
    int nwholefield=0;
    double sref=0, lfref=0, lref=0, xref=0, yref=0, zref=0;
  } force;

  struct GclCic {
    int gcl=0;
    double cic1=0,cic2=0,cic3=0,cic4=0,cic5=0;
  } gcl_cic;

  struct Connect0 {
    int connect_point=0;
    std::string conpointname;
    int connect_order=0;
    double dis_tao=0;
  } connect_0;

  // 控制回显（对应 OUT_INPUT_NAMELIST）
  bool echo_input_namelist = false;

  void load_namelist_file(const std::string& path);

  // Fortran 逻辑修正
  void postprocess() {
    
  }

  // 打印（用于回显）
  void print_summary() const;
};

} // namespace core
} // namespace orion
