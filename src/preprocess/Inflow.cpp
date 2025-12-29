#include "preprocess/Inflow.hpp"
#include "core/Runtime.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace orion::preprocess {

static constexpr double PAI = 3.14159265358979323846;

// ==========================================
// 完美气体模型初始化 (对应 Fortran: read_perfect_gasmodel)
// ==========================================
static void setup_perfect_gas_air(orion::core::Params::Inflow& F) {
    // 对应 Fortran: if( gasmodel == 'air.dat') then ...
    
    // 设置组分数量 ns = 2
    F.ns = 2;

    // 分配内存
    F.ws.resize(2);
    F.cn_init.resize(2);
    F.ws1.resize(2);
    F.ms.resize(2);
    F.ms1.resize(2);

    // 设置分子量 (Fortran: ws(1)=28.0, ws(2)=32.0)
    // 注意：这里先设为 28.0/32.0，稍后在 init_inflow 主逻辑中会乘以 1.0e-3
    F.ws[0] = 28.0; // N2
    F.ws[1] = 32.0; // O2

    // 设置初始浓度 (Fortran: cn_init(1)=0.79, cn_init(2)=0.21)
    F.cn_init[0] = 0.79; // N2
    F.cn_init[1] = 0.21; // O2

    // 设置气体常数
    F.gama = 1.40;
    F.prl  = 0.72;
    F.prt  = 0.90;

    // 打印信息 (仅主核)
    if (orion::core::Runtime::is_root()) {
        std::cout << "[GasModel] Loaded Perfect Gas (Air):\n";
        std::cout << "  Species: N2 (79%), O2 (21%)\n";
        std::cout << "  Gamma: " << F.gama << ", PrL: " << F.prl << ", PrT: " << F.prt << "\n";
    }
}

// ==========================================
// Standard Atmosphere Model (Fortran port)
// ==========================================
// 对应 Fortran: subroutine air(h1,t,p,den,a)
void call_air(double h_km, double& t, double& p, double& den, double& a, double gama = 1.4, double rjmk = 287.053) {
    
    double h = h_km * 1000.0; // km -> m
    double r = 287.053;
    double g0 = 9.80665;
    double rp = 6.37111e6;
    double g = std::pow(rp / (rp + h), 2) * g0;

    // Standard Atmosphere Constants
    double t0 = 288.15;
    double p0 = 10.1325e2; 
    double rho0 = 1.225;

    double t11 = 216.65;
    double p11 = 2.2632e2;
    double rho11 = 3.6392e-1;

    double t20 = t11;
    double p20 = 5.4747e1;
    double rho20 = 8.8035e-2;

    double t32 = 228.65;
    double p32 = 8.6789;
    double rho32 = 1.3225e-2;

    double t47 = 270.65;
    double p47 = 1.1090;
    double rho47 = 1.4275e-3;

    double t52 = t47;
    double p52 = 5.8997e-1;
    double rho52 = 7.5943e-4;

    double t61 = 252.65;
    double p61 = 1.8209e-1;
    double rho61 = 2.5109e-4;

    double t79 = 180.65;
    double p79 = 1.0376e-2;
    double rho79 = 2.0010e-5;

    double t90 = t79;
    double p90 = 1.6437e-3;
    double rho90 = 3.4165e-6;

    double t100 = 210.02;
    double p100 = 3.0070e-4;
    double rho100 = 5.6044e-7;

    double t110 = 257.00;
    double p110 = 7.3527e-5;
    double rho110 = 9.7081e-8;

    double t120 = 349.49;
    double p120 = 2.5209e-5;
    double rho120 = 2.2222e-8;

    double t150 = 892.79;
    double p150 = 5.0599e-6;
    double rho150 = 2.0752e-9;

    double t160 = 1022.20;
    double p160 = 3.6929e-6;
    double rho160 = 1.2336e-9;

    double t170 = 1103.40;
    double p170 = 2.7915e-6;
    double rho170 = 7.8155e-10;

    double t190 = 1205.40;
    double p190 = 1.6845e-6;
    double rho190 = 3.5807e-10;

    double t230 = 1322.30;
    double p230 = 6.7138e-7;
    double rho230 = 1.5640e-10;

    double t300 = 1432.10;
    double p300 = 1.8828e-7;
    double rho300 = 1.9159e-11;

    double t400 = 1487.40;
    double p400 = 4.0278e-8;
    double rho400 = 2.8028e-12;

    double t500 = 1499.20;
    double p500 = 1.0949e-8;
    double rho500 = 5.2148e-13;

    double t600 = 1506.10;
    double p600 = 3.4475e-9;
    double rho600 = 1.1367e-13;

    double t700 = 1507.60;
    double p700 = 1.1908e-9;
    double rho700 = 1.5270e-13;

    double rho = 0.0;

    // Layer calculation
    if (h <= 11019.0) {
        double al1 = (t11 - t0) / 11019.0;
        t = t0 + al1 * h;
        p = p0 * std::pow(t / t0, -g / (r * al1));
        rho = rho0 * std::pow(t / t0, -1.0 - g / (r * al1));
    } else if (h <= 20063.0) {
        t = t11;
        p = p11 * std::exp(-g * (h - 11019.0) / (r * t11));
        rho = rho11 * std::exp(-g * (h - 11019.0) / (r * t11));
    } else if (h <= 32162.0) {
        double al2 = (t32 - t20) / (32162.0 - 20063.0);
        t = t11 + al2 * (h - 20063.0);
        p = p20 * std::pow(t / t11, -g / (r * al2));
        rho = rho20 * std::pow(t / t11, -1.0 - g / (r * al2));
    } else if (h <= 47350.0) {
        double al3 = (t47 - t32) / (47350.0 - 32162.0);
        t = t32 + al3 * (h - 32162.0);
        p = p32 * std::pow(t / t32, -g / (r * al3));
        rho = rho32 * std::pow(t / t32, -1.0 - g / (r * al3));
    } else if (h <= 52429.0) {
        t = t47;
        p = p47 * std::exp(-g * (h - 47350.0) / (r * t47));
        rho = rho47 * std::exp(-g * (h - 47350.0) / (r * t47));
    } else if (h <= 61591.0) {
        double al4 = (t61 - t52) / (61591.0 - 52429.0);
        t = t47 + al4 * (h - 52429.0);
        p = p52 * std::pow(t / t47, -g / (r * al4));
        rho = rho52 * std::pow(t / t47, -1.0 - g / (r * al4));
    } else if (h <= 79994.0) {
        double al5 = (t79 - t61) / (79994.0 - 61591.0);
        t = t61 + al5 * (h - 61591.0);
        p = p61 * std::pow(t / t61, -g / (r * al5));
        rho = rho61 * std::pow(t / t61, -1.0 - g / (r * al5));
    } else if (h <= 90000.0) {
        t = t79;
        p = p79 * std::exp(-g * (h - 79994.0) / (r * t79));
        rho = rho79 * std::exp(-g * (h - 79994.0) / (r * t79));
    } else if (h <= 100000.0) {
        double al6 = (t100 - t90) / 10000.0;
        t = t79 + al6 * (h - 90000.0);
        p = p90 * std::pow(t / t79, -g / (r * al6));
        rho = rho90 * std::pow(t / t79, -1.0 - g / (r * al6));
    } else if (h <= 110000.0) {
        double al7 = (t110 - t100) / 10000.0;
        t = t100 + al7 * (h - 100000.0);
        p = p100 * std::pow(t / t100, -g / (r * al7));
        rho = rho100 * std::pow(t / t100, -1.0 - g / (r * al7));
    } else if (h <= 120000.0) {
        double al8 = (t120 - t110) / 10000.0;
        t = t110 + al8 * (h - 110000.0);
        p = p110 * std::pow(t / t110, -g / (r * al8));
        rho = rho110 * std::pow(t / t110, -1.0 - g / (r * al8));
    } else if (h <= 150000.0) {
        double al9 = (t150 - t120) / 30000.0;
        t = t120 + al9 * (h - 120000.0);
        p = p120 * std::pow(t / t120, -g / (r * al9));
        rho = rho120 * std::pow(t / t120, -1.0 - g / (r * al9));
    } else if (h <= 160000.0) {
        double al10 = (t160 - t150) / 10000.0;
        t = t150 + al10 * (h - 150000.0);
        p = p150 * std::pow(t / t150, -g / (r * al10));
        rho = rho150 * std::pow(t / t150, -1.0 - g / (r * al10));
    } else if (h <= 170000.0) {
        double al11 = (t170 - t160) / 10000.0;
        t = t160 + al11 * (h - 160000.0);
        p = p160 * std::pow(t / t160, -g / (r * al11));
        rho = rho160 * std::pow(t / t160, -1.0 - g / (r * al11));
    } else if (h <= 190000.0) {
        double al12 = (t190 - t170) / 20000.0;
        t = t170 + al12 * (h - 170000.0);
        p = p170 * std::pow(t / t170, -g / (r * al12));
        rho = rho170 * std::pow(t / t170, -1.0 - g / (r * al12));
    } else if (h <= 230000.0) {
        double al13 = (t230 - t190) / 40000.0;
        t = t190 + al13 * (h - 190000.0);
        p = p190 * std::pow(t / t190, -g / (r * al13));
        rho = rho190 * std::pow(t / t190, -1.0 - g / (r * al13));
    } else if (h <= 300000.0) {
        double al14 = (t300 - t230) / 70000.0;
        t = t230 + al14 * (h - 230000.0);
        p = p230 * std::pow(t / t230, -g / (r * al14));
        rho = rho230 * std::pow(t / t230, -1.0 - g / (r * al14));
    } else if (h <= 400000.0) {
        double al15 = (t400 - t300) / 100000.0;
        t = t300 + al15 * (h - 300000.0);
        p = p300 * std::pow(t / t300, -g / (r * al15));
        rho = rho300 * std::pow(t / t300, -1.0 - g / (r * al15));
    } else if (h <= 500000.0) {
        double al16 = (t500 - t400) / 100000.0;
        t = t400 + al16 * (h - 400000.0);
        p = p400 * std::pow(t / t400, -g / (r * al16));
        rho = rho400 * std::pow(t / t400, -1.0 - g / (r * al16));
    } else if (h <= 600000.0) {
        double al17 = (t600 - t500) / 100000.0;
        t = t500 + al17 * (h - 500000.0);
        p = p500 * std::pow(t / t500, -g / (r * al17));
        rho = rho500 * std::pow(t / t500, -1.0 - g / (r * al17));
    } else if (h <= 700000.0) {
        double al18 = (t700 - t600) / 100000.0;
        t = t600 + al18 * (h - 600000.0);
        p = p600 * std::pow(t / t600, -g / (r * al18));
        rho = rho600 * std::pow(t / t600, -1.0 - g / (r * al18));
    }

    a = std::sqrt(1.4 * r * t);
    p = p * 100.0; // Convert to Pa
    den = rho;
}

// ==========================================
// Stagnation Params (Fortran port)
// ==========================================
// 对应 Fortran: subroutine pr_stag
void call_pr_stag(orion::core::Params& params) {
    auto& F = params.inflow;
    
    double m2 = F.moo * F.moo;
    double gam1 = F.gama - 1.0;
    double ogm1 = 1.0 / gam1;
    double gam2 = F.gama + 1.0;
    double arg;

    double pst_val = 0.0;
    double rst_val = 0.0;

    if (F.moo >= 1.0) {
        arg = 1.0 + 0.5 * gam1 * m2;
        double term1 = std::pow(0.5 * gam2 * m2, F.gama * ogm1);
        double term2 = std::pow(2.0 * F.gama * m2 / gam2 - gam1 / gam2, ogm1);
        pst_val = F.poo * term1 / term2;
        rst_val = F.gama * m2 * pst_val / (F.too * arg);
    } else {
        arg = 1.0 + 0.5 * gam1 * m2;
        pst_val = F.poo * std::pow(arg, F.gama * ogm1);
        rst_val = F.roo * std::pow(arg, ogm1);
    }
    
    if (orion::core::Runtime::is_root()) {
        std::cout << "[Info] Stagnation P: " << pst_val << ", Rho: " << rst_val << "\n";
    }
}

// ==========================================
// Main Function: init_inflow
// ==========================================

void init_inflow(orion::core::Params& params, FlowFieldSet& fs) {
    auto& F = params.inflow; 

    // [Step 0] 默认使用完美空气模型 (如果未指定)
    // 对应 Fortran: read_perfect_gasmodel
    if (F.ns <= 1) {
        setup_perfect_gas_air(F);
    }

    // 1. Convert Angles to Radians
    F.attack   = F.attack * PAI / 180.0;
    F.sideslip = F.sideslip * PAI / 180.0;

    // 2. Direction Cosines
    F.uoo = std::cos(F.attack) * std::cos(F.sideslip);
    F.voo = std::sin(F.attack) * std::cos(F.sideslip);
    F.woo = std::sin(F.sideslip);

    // 3. Non-dimensional Base
    F.roo = 1.0;
    F.too = 1.0;

    // 4. Reynolds Scaling
    if (std::abs(F.lfref) > 1e-30) {
        F.reynolds = (F.reynolds / F.lfref) * F.lref;
    }

    // 5. Species Handling (Mixed Parameters)
    int ns = F.ns;
    // Safety check for arrays
    if (F.ws.size() != (size_t)ns) F.ws.resize(ns, 28.0);
    if (F.ws1.size() != (size_t)ns) F.ws1.resize(ns);
    if (F.ms.size() != (size_t)ns) F.ms.resize(ns);
    if (F.ms1.size() != (size_t)ns) F.ms1.resize(ns);
    if (F.cn_init.size() != (size_t)ns) F.cn_init.resize(ns, 1.0/ns);

    // Fortran: do is=1,ns; ws(is) = ws(is) * 1.0e-3
    for (int is = 0; is < ns; ++is) {
        F.ws[is] *= 1.0e-3; 
    }

    // ws1(is) = 1.0/ws(is)
    for (int is = 0; is < ns; ++is) {
        F.ws1[is] = 1.0 / F.ws[is];
    }

    // mref1 = sum( cn_init * ws1 )
    F.mref1 = 0.0;
    for (int is = 0; is < ns; ++is) {
        double cn = F.cn_init[is];
        F.mref1 += cn * F.ws1[is];
    }

    if (F.mref1 != 0.0) F.mref = 1.0 / F.mref1;

    for (int is = 0; is < ns; ++is) {
        F.ms[is] = F.ws[is] * F.mref1;
        F.ms1[is] = 1.0 / F.ms[is];
    }

    // 6. Atmosphere Calculation
    if (F.height < 0.0000) {
        // User Inputs Check (Custom conditions)
        bool p_given = (F.pref > 0.0);
        bool t_given = (F.tref > 0.0);
        bool r_given = (F.rref > 0.0);

        if (p_given && t_given && r_given) {
            std::cerr << "Error: Pressure, Temperature, and Density cannot be specified simultaneously.\n";
            std::exit(-1);
        }
        if (!p_given && !r_given) {
             std::cerr << "Error: Pressure and Density cannot be both zero/negative.\n";
        }

        double R_spec = F.rjmk * F.mref1; 

        if (p_given && t_given && !r_given) {
            F.rref = F.pref / (F.tref * R_spec);
        } else if (r_given && t_given && !p_given) {
            F.pref = F.rref * F.tref * R_spec;
        } else if (p_given && r_given && !t_given) {
            F.tref = F.pref / (F.rref * R_spec);
        } else {
             std::cerr << "Error: Invalid combination of Pressure, Temperature, and Density.\n";
             std::exit(-1);
        }

        F.ccon = (1.0 + 110.4/273.15) / (1.0 + 110.4/F.tref);
        F.visloo = std::sqrt(F.tref/273.15) * F.ccon * 1.715e-5;
        F.ccoo = std::sqrt(F.gama * F.pref / F.rref);

    } else {
        // Standard Atmosphere
        call_air(F.height, F.tref, F.pref, F.rref, F.ccoo, F.gama, F.rjmk);
        
        // Recalculate to ensure consistency with current R_spec
        double R_spec = F.rjmk * F.mref1;
        F.rref = F.pref / (F.tref * R_spec);

        F.ccon = (1.0 + 110.4/273.15) / (1.0 + 110.4/F.tref);
        F.visloo = std::sqrt(F.tref/273.15) * F.ccon * 1.715e-5;
    }

    // 7. Reference Values & Non-dimensionalization
    F.vref = F.ccoo * F.moo;

    if (F.height >= 0.0) {
        F.reynolds = F.rref * F.vref * F.lref / F.visloo;
    }

    F.poo = 1.0 / (F.gama * F.moo * F.moo);
    
    double v2 = F.uoo*F.uoo + F.voo*F.voo + F.woo*F.woo;
    F.eoo = F.poo / (F.gama - 1.0) + 0.5 * F.roo * v2;
    F.hoo = (F.eoo + F.poo) / F.roo;

    // Print Atmosphere (Root only)
    if (orion::core::Runtime::is_root()) {
        std::cout << "[Inflow] Atmospheric Conditions\n";
        std::cout << std::scientific << std::setprecision(5);
        std::cout << "  - Height (km)    : " << F.height << "\n";
        std::cout << "  - Temperature (K): " << F.tref << "\n";
        std::cout << "  - Pressure (Pa)  : " << F.pref << "\n";
        std::cout << "  - Density (kg/m3): " << F.rref << "\n";
        std::cout << "  - SpeedOfSound   : " << F.ccoo << "\n";
        std::cout << "  - Reynolds       : " << F.reynolds << "\n";
        std::cout << "  - RefViscosity   : " << F.visloo << "\n";
        std::cout << "  - Mach           : " << F.moo << "\n";
        std::cout << "  - RefVelocity.   : " << F.vref << "\n";
        std::cout << "  - RefLengthFlight: " << F.lfref << "\n";
        std::cout << "  - GridLengthRef. : " << F.lref << "\n";
        std::cout << "================================================\n";
    }

    // 8. Initial Conservation Variables
    F.q1oo = F.roo;
    F.q2oo = F.roo * F.uoo;
    F.q3oo = F.roo * F.voo;
    F.q4oo = F.roo * F.woo;
    F.q5oo = F.eoo;

    // 9. Stagnation Point Params
    call_pr_stag(params);

    // 10. Coefficients
    double gm1_m2 = (F.gama - 1.0) * F.moo * F.moo;
    if (gm1_m2 != 0.0) {
        F.cq_lam = 1.0 / (gm1_m2 * F.prl);
        F.cq_tur = 1.0 / (gm1_m2 * F.prt);
    }
    
    if (F.tref != 0.0) F.visc = 110.4 / F.tref;
    if (F.reynolds != 0.0) F.re = 1.0 / F.reynolds;

    // 11. Allocate Residual Arrays in FlowFieldSet
    size_t num_blocks = fs.blocks.size();
    fs.residual_norm.resize(num_blocks, 0.0);
    fs.residual_max.resize(num_blocks, 0.0);
}

} // namespace orion::preprocess