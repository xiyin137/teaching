(*
  Conformal Blocks and Bootstrap Demo
  ====================================

  Background
  ----------
  In a conformal field theory (CFT), the operator product expansion (OPE)
  decomposes four-point functions into conformal blocks — kinematically
  determined functions that encode the contribution of each conformal
  family.  Crossing symmetry (the associativity constraint on the OPE)
  yields nontrivial sum rules, which form the basis of the conformal
  bootstrap program.

  This script computes conformal blocks and crossing-equation quantities
  in three spacetime dimensions:

    Section 1 -- 2D:  Global blocks factorize holomorphically.
      G_{Delta,ell}(z,zbar) = k_{Delta+ell}(z)*k_{Delta-ell}(zbar) + (z<->zbar).
      We compute P2D crossing coefficients and gap-scan tables.

    Section 2 -- 4D:  Blocks involve the Dolan-Osborn formula with a 1/(z-zbar)
      prefactor and satisfy the conformal Casimir equation.  We verify the
      Casimir eigenvalue numerically and compute crossing-derivative vectors
      for the bootstrap.

    Section 3 -- 3D:  Delegates to the Python/PyCFTBoot runner for SDPB-based
      numerical bootstrap bounds.

  Run with:
    math -script "Conformal blocks and bootstrap demo streamlined.wl"

  Parameter glossary
  ------------------
  Delta, delta   conformal dimension of an exchanged operator
  spin, ell      spin of an exchanged operator
  z, zbar        cross-ratio variables (complex in general)
  u, v           u = z*zbar, v = (1-z)*(1-zbar)
  delta1         external scalar dimension (identical scalars: Delta_phi)
  d1..d4         external dimensions in the general 4D Casimir operator

  References
  ----------
  - Dolan, Osborn, hep-th/0309180 (4D conformal blocks)
  - Rattazzi, Rychkov, Tonni, Vichi (RRTV), 0807.0004 (bootstrap bounds)
  - Simmons-Duffin, 1502.02033 (bootstrap review)
*)

ClearAll["Global`*"];

outDir = FileNameJoin[{Directory[], "streamlined_output_main_wl"}];
If[! DirectoryQ[outDir], CreateDirectory[outDir, CreateIntermediateDirectories -> True]];

(* ===================== 2D section ===================== *)
(* In 2D the global conformal group is SL(2,R) x SL(2,R), so blocks
   factorize into a product of holomorphic and anti-holomorphic SL(2,R)
   blocks k_beta(x).  The crossing equation for identical scalars
   with dimension delta1 reads:
     v^{delta1} G(z,zbar) = u^{delta1} G(1-z,1-zbar).
   We extract the P2D crossing coefficients by Taylor-expanding the
   crossing vector around z = zbar = 1/2 (the crossing-symmetric point). *)

(* SL(2,R) block: k_beta(x) = x^{beta/2} _2F_1(beta/2, beta/2; beta; x). *)
kBeta[beta_, x_] := x^(beta/2) Hypergeometric2F1[beta/2, beta/2, beta, x];

(* 2D global block with explicit _2F_1 (no k_beta shorthand here).
   Equivalent to kBeta[delta+spin, z] * kBeta[delta-spin, zbar]. *)
Block2D[delta_, spin_, z_, zbar_] := (
  z^((delta + spin)/2) *
    zbar^((delta - spin)/2) *
    Hypergeometric2F1[(delta + spin)/2, (delta + spin)/2, delta + spin, z] *
    Hypergeometric2F1[(delta - spin)/2, (delta - spin)/2, delta - spin, zbar]
);

(* Crossing block for identical external scalars with dimension delta1:
   F = v^{delta1} G(z,zbar) - u^{delta1} G(1-z,1-zbar).
   The bootstrap equation is: identity + Sum_i OPE_i^2 * F_i = 0. *)
CrossingBlock2D[delta1_, delta_, spin_, z_, zbar_] :=
  (1 - z)^delta1 (1 - zbar)^delta1 Block2D[delta, spin, z, zbar] -
   z^delta1 zbar^delta1 Block2D[delta, spin, 1 - z, 1 - zbar];

(* P2D extracts the x^1 and x^3 Taylor coefficients of the crossing block
   evaluated on the diagonal z = zbar = 1/2 + x.  These two components form
   the "crossing derivative vector" in the simplest 2D truncation used by
   RRTV (0807.0004) to derive bootstrap bounds. *)
P2D[delta1_, delta_, spin_] := Module[{series},
  series = Normal @ Series[CrossingBlock2D[delta1, delta, spin, 1/2 + x, 1/2 + x], {x, 0, 3}];
  {Coefficient[series, x, 1], Coefficient[series, x, 3]}
];

(* Sample P2D table: delta1 = 1/8 (2D Ising), spin = 0, weights 0..6.
   Reproduces the notebook's Table 1 entries for cross-checking. *)
sample2D = Table[
  With[{p = N[P2D[1/8, weight, 0], 40]}, {weight, p[[1]], p[[2]]}],
  {weight, 0, 6, 1}
];

Export[
  FileNameJoin[{outDir, "p2d_spin0_table_wl.csv"}],
  Prepend[sample2D, {"weight", "p2d_x", "p2d_x3"}],
  "CSV"
];

(* Gap scan: for each spin, sweep operator dimensions starting from the gap
   (or the unitarity bound, whichever is larger).  The rescaling by
   1.1^spin * 2^weight improves visibility across orders of magnitude in
   the crossing coefficients -- this is the same normalization used in
   the notebook's gap-dependence plots. *)
GapScanRows[delta1_, gap_] := Module[{rows = {}, p, scale, wMin},
  Do[
    wMin = If[spin == 0, Max[spin, gap], spin];
    Do[
      p = N[P2D[delta1, weight, spin], 40];
      scale = (1.1^spin) 2^weight;
      AppendTo[rows, {spin, weight, p[[1]]/scale, p[[2]]/scale}],
      {weight, wMin, 8, 0.5}
    ],
    {spin, {0, 2, 4}}
  ];
  rows
];

Do[
  Export[
    FileNameJoin[{outDir, "p2d_gap_scan_gap" <> ToString[gap] <> "_wl.csv"}],
    Prepend[GapScanRows[1/8, gap], {"spin", "weight", "p2d_x_scaled", "p2d_x3_scaled"}],
    "CSV"
  ],
  {gap, {0, 1, 2}}
];

(* Diagonal crossing residual with a toy truncated spectrum.
   Evaluates identity + Sum OPE^2 * F at z = zbar = x.
   The residual should be small near x = 1/2 if the spectrum is reasonable. *)
ToyResidual2D[x_] := Module[{delta1 = 0.125, res},
  res = (1 - x)^(2 delta1) - x^(2 delta1);
  res += 0.24 CrossingBlock2D[delta1, 1.4126, 0, x, x];
  res += 0.016 CrossingBlock2D[delta1, 3.8303, 2, x, x];
  res += 0.002 CrossingBlock2D[delta1, 5.5, 4, x, x];
  N[res, 40]
];

diag2D = Table[{x, Re[ToyResidual2D[x]], Abs[ToyResidual2D[x]]}, {x, 0.05, 0.45, 0.01}];
Export[
  FileNameJoin[{outDir, "crossing_2d_diagonal_wl.csv"}],
  Prepend[diag2D, {"x", "residual_real", "residual_abs"}],
  "CSV"
];

(* ===================== 4D section ===================== *)
(* In 4D, conformal blocks for external scalars take the Dolan-Osborn form
   (hep-th/0309180).  The block G_{Delta,ell}(z,zbar) involves a 1/(z-zbar)
   prefactor and satisfies the conformal Casimir equation:
     DD[G] = (1/2)(Delta(Delta-4) + ell(ell+2)) G.
   Below we define the DD differential operator, verify the Casimir equation
   numerically, and compute crossing-derivative vectors for the bootstrap. *)

(* DD: the 4D conformal Casimir differential operator for general external
   dimensions d1..d4.  Acts on functions of (z, zbar).
   The identity DD[f/(z-zbar)] = (DD0[f] + 2f)/(z-zbar) (checked below)
   lets one factor out the 1/(z-zbar) pole when verifying Casimir. *)
DDExpr[expr_, d1_, d2_, d3_, d4_] :=
  z^2 (1 - z) D[expr, {z, 2}] +
   (((d1 - d2 - d3 + d4)/2) - 1) z^2 D[expr, z] +
   ((d1 - d2) (d3 - d4)/4) z expr +
   2 z zbar/(z - zbar) (1 - z) D[expr, z] +
   zbar^2 (1 - zbar) D[expr, {zbar, 2}] +
   (((d1 - d2 - d3 + d4)/2) - 1) zbar^2 D[expr, zbar] +
   ((d1 - d2) (d3 - d4)/4) zbar expr +
   2 z zbar/(zbar - z) (1 - zbar) D[expr, zbar];

(* DD0: auxiliary Casimir operator that acts on the numerator f when the
   block is written as f/(z-zbar).  Related to DD by:
   DD[f/(z-zbar)] = (DD0[f] + 2f)/(z-zbar). *)
DD0Expr[expr_, d1_, d2_, d3_, d4_] :=
  z^2 (1 - z) D[expr, {z, 2}] +
   (((d1 - d2 - d3 + d4)/2) + 1) z^2 D[expr, z] - 2 z D[expr, z] +
   ((d1 - d2 + 2) (d3 - d4 - 2)/4) z expr +
   zbar^2 (1 - zbar) D[expr, {zbar, 2}] +
   (((d1 - d2 - d3 + d4)/2) + 1) zbar^2 D[expr, zbar] - 2 zbar D[expr, zbar] +
   ((d1 - d2 + 2) (d3 - d4 - 2)/4) zbar expr;

(* Verify the operator identity DD[f/(z-zbar)] = (DD0[f] + 2f)/(z-zbar)
   symbolically.  The result should simplify to 0 for arbitrary f(z,zbar). *)
relationCheck4D = Simplify[
  DDExpr[f[z, zbar]/(z - zbar), d1, d2, d3, d4] -
   (DD0Expr[f[z, zbar], d1, d2, d3, d4] + 2 f[z, zbar])/(z - zbar)
];

(* 4D scalar conformal block (Dolan-Osborn, hep-th/0309180 Eq. 2.20).
   G_{Delta,ell}(z,zbar) = [z^{h+1} zbar^{hbar} F(h,h;2h;z) F(hbar-1,hbar-1;2hbar-2;zbar)
                            - (z <-> zbar)] / (z - zbar),
   where h = (Delta+ell)/2, hbar = (Delta-ell)/2.
   The 1/(z-zbar) prefactor is characteristic of even-dimensional blocks. *)
Block4D[delta_, spin_, z_, zbar_] := (
  z^((delta + spin)/2 + 1) *
    zbar^((delta - spin)/2)/(z - zbar) *
    Hypergeometric2F1[(delta + spin)/2, (delta + spin)/2, delta + spin, z] *
    Hypergeometric2F1[(delta - spin)/2 - 1, (delta - spin)/2 - 1, delta - spin - 2, zbar]
  + zbar^((delta + spin)/2 + 1) *
    z^((delta - spin)/2)/(zbar - z) *
    Hypergeometric2F1[(delta + spin)/2, (delta + spin)/2, delta + spin, zbar] *
    Hypergeometric2F1[(delta - spin)/2 - 1, (delta - spin)/2 - 1, delta - spin - 2, z]
);

(* Numeric check of the Casimir equation: DD[G] - eigenvalue * G.
   The eigenvalue for a spin-ell primary of dimension Delta in 4D is
   (1/2)(Delta(Delta-4) + ell(ell+2)).  A correct block implementation
   should give residual ~ 0 to working precision. *)
CasimirResidual4D[delta_, spin_, z0_, zbar0_] := N[
  (
    DDExpr[Block4D[delta, spin, z, zbar], 2, 2, 2, 2] -
     1/2 (delta (delta - 4) + spin (spin + 2)) Block4D[delta, spin, z, zbar]
  ) /. {z -> z0, zbar -> zbar0},
  40
];

(* Test the Casimir equation at two generic complex points (Delta,ell):
   (3,0) and (5,2).  Using complex z, zbar avoids accidental cancellations. *)
casimirRows = {
  {3, 0, 0.4 + 0.5 I, 0.5 - 0.3 I, CasimirResidual4D[3, 0, 0.4 + 0.5 I, 0.5 - 0.3 I]},
  {5, 2, 0.3 + 0.2 I, 0.45 - 0.15 I, CasimirResidual4D[5, 2, 0.3 + 0.2 I, 0.45 - 0.15 I]}
};

Export[
  FileNameJoin[{outDir, "casimir_checks_4d_wl.csv"}],
  Prepend[
    ({#[[1]], #[[2]], #[[3]], #[[4]], Re[#[[5]]], Im[#[[5]]], Abs[#[[5]]]} & /@ casimirRows),
    {"delta", "spin", "z", "zbar", "residual_real", "residual_imag", "residual_abs"}
  ],
  "CSV"
];

(* 4D crossing block: (z-zbar) * [v^{delta1} G(z,zbar) - u^{delta1} G(1-z,1-zbar)].
   The (z-zbar) prefactor cancels the 1/(z-zbar) pole in the block, leaving
   a polynomial-like object suitable for Taylor expansion. *)
CrossingBlock4D[delta1_, delta_, spin_, z_, zbar_] :=
  Simplify[
    (z - zbar) ((1 - z)^delta1 (1 - zbar)^delta1 Block4D[delta, spin, z, zbar] -
      z^delta1 zbar^delta1 Block4D[delta, spin, 1 - z, 1 - zbar])
  ];

(* Crossing derivative vector: expand CrossingBlock4D around the crossing-
   symmetric point z = zbar = 1/2 and extract Taylor coefficients F_{m,n}.
   These form the vector that enters the bootstrap linear/semidefinite program.
   The epsilon trick regularizes the expansion: substitute z = 1/2 + eps*x,
   zbar = 1/2 + eps*xbar, expand in eps, then set eps = 1.
   We keep only the bootstrap-relevant components:
   m = 0..2*deg, n = m+2..2*deg-m (step 2), which encode odd-parity
   derivatives under z <-> zbar exchange. *)
CrossingDeriv4D[delta1_, delta_, spin_, deg_] := Module[{poly, mat, rows = {}},
  mat = CoefficientList[
    (((CrossingBlock4D[delta1, delta, spin, 1/2 + eps x, 1/2 + eps xbar] +
       O[eps]^(2 deg + 1)) // Normal // N) /. eps -> 1),
    {x, xbar}
  ];
  Do[
    AppendTo[
      rows,
      {
        m,
        n,
        If[m + 1 <= Length[mat] && n + 1 <= Length[mat[[m + 1]]], mat[[m + 1, n + 1]], 0]
      }
    ],
    {m, 0, 2 deg}, {n, m + 2, 2 deg - m, 2}
  ];
  rows
];

sampleDeriv4D = Join[
  ({3, 0, #[[1]], #[[2]], Re[#[[3]]], Im[#[[3]]]} & /@ CrossingDeriv4D[1, 3, 0, 3]),
  ({5, 2, #[[1]], #[[2]], Re[#[[3]]], Im[#[[3]]]} & /@ CrossingDeriv4D[1, 5, 2, 3])
];

Export[
  FileNameJoin[{outDir, "crossing_derivatives_4d_wl.csv"}],
  Prepend[sampleDeriv4D, {"delta", "spin", "m", "n", "coeff_real", "coeff_imag"}],
  "CSV"
];

(* ===================== 3D section ===================== *)
(* In 3D, conformal blocks do not have closed-form _2F_1 expressions.
   They are computed numerically via the Zamolodchikov-like recursion
   implemented in PyCFTBoot, which then feeds into SDPB (a semidefinite
   program solver) to derive rigorous bootstrap bounds.  This section
   simply records the delegation to the Python driver. *)

notes = {
  "The original notebook's 3D recursion and SDPB parts are heavy by design.",
  "This streamlined file keeps 2D/4D derivations explicit and delegates",
  "3D production runs to the Python script for reliability and speed.",
  "For production runs, use the Python driver in this repo:",
  "  python3 conformal_blocks_and_bootstrap_demo_streamlined.py 3d --delta-sigma 0.518",
  "or the bootstrap workflow under /path/to/teaching/Bootstrap/ising3d."
};

Export[FileNameJoin[{outDir, "three_d_section_notes.txt"}], StringRiffle[notes, "\n"], "Text"];

(* Summary *)
summaryText = StringRiffle[
  {
    "Wrote 2D outputs: p2d tables, gap scans, diagonal residual.",
    "Wrote 4D outputs: DD/DD0 relation check, Casimir checks, derivative vectors.",
    "Wrote 3D note with production command path.",
    "Output directory: " <> outDir
  },
  "\n"
];

Export[FileNameJoin[{outDir, "SUMMARY.txt"}], summaryText, "Text"];
Print[summaryText];
