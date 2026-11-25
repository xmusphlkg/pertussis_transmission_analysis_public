# Model and analysis equations

Equation numbers follow the supplementary methods. The state transitions and parameters are tabulated separately. [Table S8](../source_data/tables/tableS8.csv)

## Vaccination at a scheduled visit

$E$ is the eligible population at the visit. Standalone intranasal uptake $c_N$ is independent of Tdap uptake $c_T$ among the eligible states. In the coadministration scenario, the same $cE$ recipients receive both products.

$$
D_{\mathrm{Tdap}}=c_TE,\qquad\text{(S1)}
$$

$$
D_{\mathrm{nasal}}=c_NE,\qquad\text{(S2)}
$$

$$
N_{\mathrm{both}}=c_Tc_NE.\qquad\text{(S3)}
$$

## Age accounting

Each model bin $a$ contributes to four completed-age groups $g$ according to month-specific weights $w_{ag}$. Its population is $N_a$. The groups cover infants younger than one year, ages 1–9 and 10–19 years, and adults aged 20 years or older.

$$
\sum_g w_{ag}(t)=1,\qquad \sum_g\sum_{a=1}^{54}w_{ag}(t)N_a(t)=\sum_{a=1}^{54}N_a(t)=N(t).\qquad\text{(S4)}
$$

## Reported cases and symptomatic incidence

$T_a$ and $M_a$ are typical and mild symptomatic onsets. $p(a)$ is the typical-course reporting probability and $r$ scales reporting for the mild course. The prospective reference diagnostic multiplier is one.

$$
R_a(t)=p(a)[T_a(t)+rM_a(t)],\qquad\text{(S5)}
$$

$$
I_a(t)=T_a(t)+M_a(t).\qquad\text{(S6)}
$$

## Transmission between age groups

$g(a)$ maps model bin $a$ to a contact group. $J_h$ is the infectious fraction in group $h$, weighted by relative mild infectiousness $\rho$ and by onward-contribution multiplier $v_j$ for nasal layer $j$. The contact matrices describe household and other contacts, with non-household contact multiplier $m(t)$. The force of infection includes seasonal forcing $s(t)$, age multiplier $c_a$ and importation $\iota(t)$. The transmission coefficient $q=R_0/[D_t\rho(C_{2019})]$ uses the typical infectious period and the spectral radius of the total 2019 contact matrix. $F_{X,aj}$ is infection flow from susceptible base state $X$ with relative susceptibility $\sigma_X$ and acquisition multiplier $u_j$. The unprotected layer has $u_0=v_0=1$.

$$
J_h(t)=\frac{\sum_{a:g(a)=h}\sum_{j=0}^{2}v_j[I_{t,aj}(t)+\rho I_{m,aj}(t)]}{N_h(t)}.\qquad\text{(S7)}
$$

$$
\lambda_a(t)=q\,s(t)c_a\sum_h\left[C^{\mathrm{home}}_{g(a)h}(t)+m(t)C^{\mathrm{other}}_{g(a)h}(t)\right]J_h(t)+\iota(t).\qquad\text{(S8)}
$$

$$
F_{X,aj}(t)=u_j\lambda_a(t)\sigma_X X_{aj}(t).\qquad\text{(S9)}
$$

## Clinical input

$R$ is the ratio of arithmetic mean bacterial-density area under the curve (AUC) in the BPZE1 and placebo groups, including zero observations. The decomposition uses the proportion with a positive AUC and the group mean among those with positive AUC. The three profile anchors are the estimate and the two limits from 20,000 within-arm bootstrap replicates. [Trial supplementary data](https://doi.org/10.1038/s41541-026-01443-7)

$$
R=\frac{\overline{\mathrm{AUC}}_{\mathrm{BPZE1}}}{\overline{\mathrm{AUC}}_{\mathrm{placebo}}}=0.0569875.\qquad\text{(S10)}
$$

$$
R=\frac{\Pr(AUC_{BPZE1}>0)}{\Pr(AUC_{placebo}>0)}\times\frac{E(AUC_{BPZE1}\mid AUC>0)}{E(AUC_{placebo}\mid AUC>0)}.\qquad\text{(S11)}
$$

## Intranasal protection and waning

The assumed translation exponent $b$ maps the observed AUC ratio to residual transmission potential at day 90. Two sequential protected stages have total mean duration $D$ years and transition rate $\gamma_D$ per day. With $t$ in days and $z=2t/(365.25D)$, $P_1$, $P_2$ and $P_W$ are the two protected fractions and the fully waned fraction. $R_P$ is the combined residual multiplier in a protected state. An allocation $a$ of 0, 0.5 or 1 partitions protection between acquisition and onward contribution. S24 gives the reference profile with $b=a=0.5$ and $D=3$.

$$
T_{90}=R^b=\exp(b\log R),\qquad\text{(S12)}
$$

$$
\gamma_D=\frac{2}{365.25D}.\qquad\text{(S13)}
$$

$$
S_D(t)=\exp\left(-\frac{2t}{365.25D}\right)\left(1+\frac{2t}{365.25D}\right).\qquad\text{(S14)}
$$

$$
P_1(t)=e^{-z},\qquad\text{(S15)}
$$

$$
P_2(t)=ze^{-z},\qquad\text{(S16)}
$$

$$
P_W(t)=1-S_D(t).\qquad\text{(S17)}
$$

$$
R_P=\frac{T_{90}-[1-S_D(90)]}{S_D(90)}.\qquad\text{(S18)}
$$

$$
[1-S_D(90)]+S_D(90)R_P=T_{90}.\qquad\text{(S19)}
$$

$$
0\leq R_P\leq1,\qquad\text{(S20)}
$$

$$
1-S_D(90)\leq T_{90}\leq1.\qquad\text{(S21)}
$$

$$
m_{\mathrm{acquisition}}=R_P^a,\qquad\text{(S22)}
$$

$$
m_{\mathrm{onward}}=R_P^{1-a}.\qquad\text{(S23)}
$$

$$
T_{90}=0.238721,\qquad R_P=0.229395,\qquad m_{\mathrm{acquisition}}=m_{\mathrm{onward}}=0.478952.\qquad\text{(S24)}
$$

## Model fitting

The 21 transformed parameters $z$ are fitted per setting. Case totals $y_i$ follow a negative binomial observation model with expected count $\mu_i$ and dispersion. Annual age counts $\mathbf n_y$ follow a Dirichlet–multinomial model conditional on observed total $N_y$, predicted age proportions and concentration. The last term is the optimisation-scale prior. Only intervals and years assigned to the fit enter the sums.

$$
\mathcal L(z)=-\sum_{i\in\mathcal I_{\mathrm{fit}}}\log f_{\mathrm{NB}}\!\left(y_i;\mu_i(z),k_i(z)\right)
-\sum_{y\in\mathcal Y_{\mathrm{fit}}}\log f_{\mathrm{DM}}\!\left(\mathbf n_y\mid N_y,\mathbf p_y(z),\kappa(z)\right)-\log p_z(z).\qquad\text{(S25)}
$$

## Matching adolescent cases

$G_A(x)$ and $G_O(x)$ are cumulative reported adolescent cases averted at protection strength $x$ under the acquisition and onward mechanisms. For each setting and horizon with two positive full-strength reductions, solve for each strength at common target $K$. Root selection uses adolescent outcomes.

$$
K=\frac{1}{2}\min\{G_A(1),G_O(1)\},\qquad\text{(S26)}
$$

$$
G_k(x_k)=K.\qquad\text{(S27)}
$$

## Health outcomes

The harmonised outcome sums infant years of life lost (YLL) and all-age years lived with disability (YLD). $w$ is the episode disability weight, $d$ its duration in years and $r$ the annual discount rate.

$$
H_{\mathrm{harmonised}}=\mathrm{YLL}_{<1}+\mathrm{YLD}_{\mathrm{all\ ages}}.\qquad\text{(S28)}
$$

$$
B_r(d)=\begin{cases}\dfrac{1-e^{-rd}}{r},&r>0,\\[6pt]d,&r=0,\end{cases}\qquad\text{(S29)}
$$

$$
YLD_{\mathrm{episode}}=wB_r(d).\qquad\text{(S30)}
$$

## Discounting administrations and health losses

$A_{s,y}$ counts actual intranasal doses in strategy $s$ and calendar year $y$. $y_0$ is the introduction year and $H$ is follow-up in years. Month $m$ belongs to follow-up year $t_m=\lfloor m/12\rfloor$. For age stratum $a$ and course $\ell$, episode counts $c$, disability weights $w$, durations $d$, deaths $q$ and life expectancy $L$ determine YLD and YLL. Annual discounting starts at introduction. Within-event discounting starts at onset or death.

$$
D_s(H;r)=\sum_{y=y_0}^{y_0+H-1}(1+r)^{-(y-y_0)}A_{s,y}.\qquad\text{(S31)}
$$

$$
B_r(u)=\int_0^u e^{-rv}\,dv=\frac{1-e^{-ru}}{r}\quad (r>0),\qquad\text{(S32)}
$$

$$
YLD_s(H)=\sum_{m=0}^{12H-1}(1+r)^{-t_m}\sum_{a,\ell}c_{sma\ell}w_\ell B_r(d_\ell),\qquad\text{(S33)}
$$

$$
YLL_s(H)=\sum_{m=0}^{12H-1}(1+r)^{-t_m}\sum_a q_{sma}B_r(L_a).\qquad\text{(S34)}
$$

## Choice of programme and per-dose price ceiling

$\Delta H_s$ is health loss averted and $\Delta C_s$ incremental cost relative to current provision. The valuation $\lambda$ is expressed per health-year averted. NHB and NMB denote net health benefit and net monetary benefit, respectively. $H_s,C_s$ and discounted actual intranasal doses $D_s$ determine value at delivered price $p$. The price boundary compares each intranasal strategy against the best standard programme in $\mathcal S$. In S39 the differences refer to that comparator. $\Delta T_s$, $\Delta V_s$ and $\Delta A_s$ are incremental treatment, standard vaccine and administration costs.

$$
E[\mathrm{NHB}_s(\lambda)]=E[\Delta H_s]-\frac{E[\Delta C_s]}{\lambda},\qquad\text{(S35)}
$$

$$
EOL_s(\lambda)=E\left[\max_j NHB_j(\lambda)-NHB_s(\lambda)\right],\qquad\text{(S36)}
$$

$$
E[\mathrm{NMB}_s(p,\lambda)]=E[\lambda H_s-C_s-pD_s].\qquad\text{(S37)}
$$

$$
p_s^*(\lambda)=\frac{E[\lambda H_s-C_s]-\max_{j\in\mathcal S}E[\lambda H_j-C_j]}{D_s}.\qquad\text{(S38)}
$$

$$
p_s^*=\frac{\lambda E[\Delta H_s]}{D_s}
-\frac{E[\Delta T_s]}{D_s}-\frac{E[\Delta V_s]}{D_s}-\frac{E[\Delta A_s]}{D_s}.\qquad\text{(S39)}
$$

## Parameter uncertainty and clinical sensitivities

EVPI measures the expected value of resolving the uncertainty represented by paired draws. For the mild-mortality sensitivity, $r_m$ is the mild-to-typical infant mortality risk ratio, $q_a(r_m)$ the recalibrated typical-course risk and the onset differences are comparator minus intervention.

$$
\mathrm{EVPI}=E\!\left[\max_s\mathrm{NMB}_s\right]-\max_s E[\mathrm{NMB}_s].\qquad\text{(S40)}
$$

$$
\Delta D(r_m)=\sum_a q_a(r_m)\left[\Delta T_a+r_m\Delta M_a\right],\qquad\text{(S41)}
$$

## Predictive checks

$y_i$ and $\hat y_i$ are observed and predicted totals in the same surveillance intervals. WAPE is weighted absolute percentage error and $Q$ compares cumulative predicted and observed cases.

$$
WAPE=100\times\frac{\sum_i|\hat y_i-y_i|}{\sum_i y_i}.\qquad\text{(S42)}
$$

$$
Q=\frac{\sum_i\hat y_i}{\sum_i y_i}.\qquad\text{(S43)}
$$
