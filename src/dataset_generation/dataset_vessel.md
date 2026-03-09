# Vessel Surrogate Model — Geometry & Variables Guide

## Overview

This document describes the geometry, variables, and boundary conditions used to build a Deep Learning surrogate model (SM) of the **vessel domain** of a PWR reactor during severe accidents. The SM replaces the coupled ICARE (core degradation) and CESAR (thermal-hydraulics) modules of the ASTEC code.

The simulations cover two accident types on a simplified 4-loop 1,300 MWe PWR:

1. **LB-LOCA** — Large Break Loss-of-Coolant Accident (with SI and CSS failure)
2. **SBO** — Station Blackout (with AFW failure)

The only source of variation across simulations is the **timing of 12 operator actions** (sampled via Sobol sequences). The reactor always starts from the same nominal-power initial condition. The SM predicts the vessel physics **up to vessel rupture**.

---

## Vessel Geometry

The vessel is a **2D axisymmetric** structure with **5 concentric rings** (columns) and **15 axial levels**, plus a single additional volume at the bottom for the **lower plenum**.

![Vessel and core geometry](vessel_and_core.png)

### Index Map

| Region | Index range | Description |
|---|---|---|
| Lower plenum | `0` | Single volume at the very bottom of the vessel |
| Vessel grid | `1 – 75` | 5 columns × 15 rows (bottom-to-top, left-to-right within each row) |
| Core subset | `11 – 46` | The first 3 columns, where fuel rods are located |
| Boundary $B_1$ | `76` | Connection point between vessel and hot leg |
| Boundary $B_2$ | `77` | Connection point between vessel and cold leg |
| Hot leg first volume ($h_1$) | `78` | First control volume of the hot leg (primary circuit) |
| Cold leg first volume ($c_1$) | `79` | First control volume of the cold leg (primary circuit) |
| Faces | `80 – 219` | 140 interfaces between adjacent vessel volumes |

### Column layout (left to right)

- **Columns 1–3** — contain fuel rods 
- **Columns 4–5** — vessel volumes without fuel

### How faces work

Each **face** sits at the interface between two volumes and carries the flow variables at that boundary. For example, face `84` is the interface between the lower plenum (index `0`) and the bottom volume of the 5th column.

![Faces geometry](faces.png)

---

## Variable Groups

### Inputs to the SM

The SM receives as input the values of the variables coming from the **hot leg** and **cold leg** to predict the values of the variables in $B_1$ and $B_2$ and in the rest of the vessel. These variables are taken from indeces $0$ for the hot leg and $12$ for the cold leg from the ASTEC vector coming from the path 'primary/volume/{variable}'.
#### Hot leg — p(h₁) (index 78) — 13 variables
P_up_primary_volume is always NaN in the hdf5 file.

| Variable | Unit |
|---|---|
| Void fraction | - |
| Steam partial pressure | Pa |
| Gas temperature | K |
| Saturation pressure | Pa |
| Hydrogen partial pressure | Pa |
| Total pressure | Pa |
| Steam mass | kg |
| Liquid density | kg/m³ |
| Liquid mass | kg |
| Saturation temperature | K |
| Void fraction of steam-water | - |
| Liquid temperature | K |

#### Cold leg — $p(c_1)$ (index 79) — 12 variables

Same 12 variables as the hot leg (listed above).

---

### Outputs of the SM

Everything below is **predicted** by the surrogate model.

#### Global variables — $s_g$ — scalar in time

| Variable | Short name | Unit |
|---|---|---|
| $H_2$ cumulated mass in the core | m cum H2 | kg |
| Corium mass in the core | m tot cor | kg |
| Total activity in domain | FP A heat | Bq |
| Maximum saturation in core meshes | sat core mesh | - |
| Mean mass flowrate of 53 fission product elements | FP (×53) | kg/s |

The 53 fission product elements are: Ac, Ag, Am, As, Ba, Br, Cd, Ce, Cm, Cs, Cu, Dy, Er, Eu, Ga, Gd, Ge, Ho, I, Ln, Kr, La, Mo, Nb, Nd, Np, Pa, Pd, Pm, Pr, Pu, Ra, Rb, Re, Rh, Ru, Sb, Se, Sm, Sn, Sr, Tb, Tc, Te, Th, Tl, Tm, U, Xe, Y, Yb, Zn, Zr.

#### Lower plenum variables — $s_p$ (index 0) — 18 variables

| Variable | Short name | Unit |
|---|---|---|
| Pressure | P | Pa |
| Gaseous phase temperature | T gas | K |
| Liquid phase temperature | T liq | K |
| Void fraction | x alpha | - |
| Saturation temperature | T sat | K |
| Hydrogen pressure | P H2 | Pa |
| Steam pressure | P steam | Pa |
| Mass of gaseous phase | m gas | kg |
| Mass of liquid phase | m liq | kg |
| Density of gaseous phase | rho gas | kg/m³ |
| Density of liquid | rho liq | kg/m³ |
| Liquid to vapor flowrate | Q liq vap | kg/s |
| Porosity of mesh with rods | porosity | - |
| Volume proportion debris classes | V deb | - |
| Volume proportion magma | V mag | - |
| Magma mass in the vessel |m magma| kg |
| Vessel debris 0 mass | m debris 0 | kg |
| Vessel debris 1 mass | m debris 1 | kg |

#### Core variables — $s_{cr}$ (indices 11–46, 36 volumes) — 4 variables per volume

| Variable | Short name | Unit |
|---|---|---|
| Fuel component temperature | T comp fuel | K |
| Clad component temperature | T comp clad | K |
| Component state of fuel | state fuel | - |
| Component state of cladding | state clad | - |

#### Vessel variables — $s_v$ (indices 1–75, 75 volumes) — 18 variables per volume

| Variable | Short name | Unit |
|---|---|---|
| Pressure | P | Pa |
| Gaseous phase temperature | T gas | K |
| Liquid phase temperature | T liq | K |
| Void fraction | x alpha | - |
| Saturation temperature | T sat | K |
| Hydrogen pressure | P H2 | Pa |
| Steam pressure | P steam | Pa |
| Mass of gaseous phase | m gas | kg |
| Mass of liquid phase | m liq | kg |
| Density of gaseous phase | rho gas | kg/m³ |
| Density of liquid | rho liq | kg/m³ |
| Liquid to vapor flowrate | Q liq vap | kg/s |
| Porosity of mesh with rods | porosity | - |
| Volume proportion debris classes | V deb | - |
| Volume proportion magma | V mag | - |
| Magma mass in the vessel |m magma| kg |
| Vessel debris 0 mass | m debris 0 | kg |
| Vessel debris 1 mass | m debris 1 | kg |

#### Face variables — $s_f$ (indices 80–219, 140 faces) — 3 variables per face

| Variable | Short name | Unit |
|---|---|---|
| Liquid mass flow rate | Q m liq | kg/s |
| Gas velocity | V gas | m/s |
| Liquid velocity | V liq | m/s |

#### Boundary $B_1$ — $s_{B_1}$ (index 76) — 3 variables
This variables comes from index $0$ of path connection/general/{variable}. The other variables in this volume are either NaN or missing in the HDF5 files.
| Variable | Short name | Unit |
|---|---|---|
| Instantaneous steam mass flow | Q steam ptv | kg/s |
| Instantaneous water flow | Q H2O ptv | kg/s |
| Cumulative total mass of water | m H2O ptv | kg |

#### Boundary $B_2$ — $s_{B_2}$ (index 77) — 3 variables
This variables comes from index $1$ of path connection/general/{variable}. The other variables in this volume are either NaN or missing in the HDF5 files.
| Variable | Short name | Unit |
|---|---|---|
| Instantaneous steam mass flow | Q steam vtp | kg/s |
| Instantaneous water flow | Q H2O vtp | kg/s |
| Cumulative total mass of water | m H2O vtp | kg |

---

## Boundary Conditions & Coupling

The vessel domain is connected to the **primary circuit** through two boundary points:

- **$B_1$ (index 76)** ↔ hot leg first volume **$h_1$ (index 78)**
- **$B_2$ (index 77)** ↔ cold leg first volume **$c_1$ (index 79)**

In the SM framework:

- **Input (from primary circuit):** at each time step, the values of $p(h_1)$ and $p(c_1)$ needs to be given to the SM to be informed about the change at the boundaries driven by the activation of the operator actions.
- **Output (predicted by SM):** all vessel, core, plenum, face, global, and boundary variables — including $s_{B_1}$ and $s_{B_2}$, which are needed to couple back to the primary circuit.

In other words, the SM takes primary-circuit conditions at $h_1$ and $c_1$ (at the vessel boundary) and predicts everything inside the vessel, plus the boundary fluxes that the primary circuit solver needs for the next coupling step.

To be more clear, so fare we have called **Inputs** those quantities that are the actual degrees of freedom of the vessel, i.e., the quantities that, if not given at a certain moment in time $t$, would make it impossible to the SM, or to any physical solver, to predict the next time step. On the other hand, **Outputs** are those quantities that are output to the model and are not really needed to predict other variables (if we remove the core variables from the dataset we can still predict the vessel ones, but if we remove $p(h_1)$ and $p(c_1)$ this task would be impossible). To be even more clear, one might devise a surrogate model that performs the mapping $(p(h_1)(t_{i+1}), p(c_1)(t_{i+1}))\rightarrow (s_{g}(t_{i+1}),s_{p}(t_{i+1}),s_{v}(t_{i+1}),s_{cr}(t_{i+1}),s_{B_1}(t_{i+1}), s_{B_2}(t_{i+1}))$. The initial condition in such mapping would not be necessary in our system as the initial condition is constant.

All this being said, depending on the surrogate model developed, one might decide to predict autoregressively the output variables at time $t_{i+1}$ from the output variables at time $t_{i}$ and the $p(h_1)(t_i)$ and $p(c_1)(t_i)$; in such case also $(s_{g}(t_i),s_{p}(t_i),s_{v}(t_i),s_{cr}(t_i),s_{B_1}(t_i), s_{B_2}(t_i))$ are input to the model, although they are not necessary for the prediction.

## SM Time-Stepping & Coupling Logic

Since ICARE and CESAR are internally coupled through a non-trivial sub-cycling scheme (with micro time-steps $\delta t_1$ and $\delta t_2$), the SM **only operates at the macro time-steps** ignoring all intermediate sub-steps.
One way to model the SM is autoregressively: at each macro step, the SM predicts the next vessel state at time $t_{i+1}$ given two inputs:

- the **current vessel state** (all vessel, core, plenum, face, global, and boundary variables) at time $t_i$;
- the **current primary circuit state** (the hot leg $h_1$ and cold leg $c_1$ variables), which is provided by the primary circuit model at time $t_i$.

By taking $p$ as input and predicting $s_{B_1}$ and $s_{B_2}$ as output, the SM **completely decouples the vessel from the rest of the reactor**. This enables a modular coupling strategy where the primary circuit can be handled by another SM or by ASTEC itself.

### Coupling loop

The vessel SM and the primary circuit model exchange data at each macro time-step as follows:

1. The **primary circuit model** computes $p(c_1)(t_i)$ and $p(h_1)(t_i)$ at time $t_i$;
2. The **vessel SM** takes $p(c_1)(t_i)$ and $p(h_1)(t_i)$ and $(s_{g}(t_i),s_{p}(t_i),s_{v}(t_i),s_{cr}(t_i),s_{B_1}(t_i), s_{B_2}(t_i))$ as input and predicts $(s_{g}(t_{i+1}),s_{p}(t_{i+1}),s_{v}(t_{i+1}),s_{cr}(t_{i+1}),s_{B_1}(t_{i+1}), s_{B_2}(t_{i+1}))$;
3. The **primary circuit model** reads the predicted boundaries $s_{B_1}(t_{i+1})$ and $s_{B_2}(t_{i+1})$ and computes $p(c_1)(t_{i+1})$ and $p(h_1)(t_{i+1})$. 
4. Repeat from step 2.
