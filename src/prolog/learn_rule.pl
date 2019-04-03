%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learn statistical classifiers by abduction
%      AUTHOR: WANG-ZHOU DAI
% ========= Main Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program is a 4-tuple: @todo: should it be modified?
% Prog = ps(MetaSubs, Signature, SizeBound, MetaRules)
%
% Signature is current program's predicates and constants
% Signature = sig([pred1/n | Preds], [Const1 | Consts]).
%
% Background knowledge is a ?-tuple
% @TODO: BK = bk(Hypothesis, Facts?, ???)
%
% Hypothesis is a list of Rules
% Hyp = hyp([Rule1, Rule2 | Rules])
%
% Each Rule is a list of atoms with an indicator of metarule
% Rule = rule([pred1, X]:-[pred2, X | U], [pred3, X | V], Metarule_name)
%
% Metarule is a template of first-order rules
% Metarule = metarule(Name, Rule_template, Pretest)
% in which the rule template is [P,X]:-[[Q,X], [R,X,Y]]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- load_foreign_library(foreign('../core/pl_data.so')),
   load_foreign_library(foreign('../core/pl_models.so')),
   load_foreign_library(foreign('../core/pl_cluster.so')).

:- expects_dialect(sicstus).
:- use_module(library(timeout)).
:- use_module(library(apply_macros)).
:- set_prolog_flag(unknown, fail).

:- ['utils.pl'].


%=========================================================================
% Abductive/inductive Proving of a label
% @Target Atoms: list form like [l_01, X, R1, R2]
% @BK: background knowledge in ps(_, _, _, _) form
% @Data: training data (pointer of DataPatch)
% @Range1: Input Data Range, indicator of instances for training
% @Prog2: Output program
% @Range2: Output Data Range, inidcator of instances covered by Prog2
%=========================================================================
learn_range([], Prog, _, Range, Prog, Range).
learn_range([Atom | Atoms], Prog1, DataPatch, Range1, Prog2, Range2):-
    % deductive prove does not change
    d_prove([Atom], Prog1, DataPatch, Range1, Range3),
    learn_range(Atoms, Prog1, DataPatch, Range3, Prog2, Range2).
learn_range([Atom | Atoms], Prog1, DataPatch, Range1, Prog2, Range2):-
    metarule(RuleName, MetaSub, (Atom:-Body), PreTest, Prog1),
    write('PRETEST BINDING: '), nl, write(PreTest), nl,
    call(PreTest),
    abduce(metasub(RuleName, MetaSub), Prog1, Prog3),
    %printprog([metasub(RuleName, MetaSub)]), nl,
    Prog3 = ps(_, _, Left, _),
    write('CLAUSES LEFT = '), write(Left), nl,
    learn_range(Body, Prog3, DataPatch, Range1, Prog4, Range3),
    learn_range(Atoms, Prog4, DataPatch, Range3, Prog2, Range2).

%=====================================================
% Deductive proving:
% d_prove(Atom, DataPatch, Program, Range1, Range2).
%
% deductive prove the @Atom in @Range1 of @DataPatch,
% and return satisfied instance indicators in @Range2
%=====================================================
d_prove([], _, _, Range, Range):-
    !.
d_prove([Atom | Atoms], Prog, DataPatch, Range1, Range2):-
    %***---primatom(Atom), !, otherwise, examples with recursive are not called
    d_prove_pred_range(Atom, Prog, DataPatch, Range1, Range3),
    d_prove(Atoms, DataPatch, Prog, Range3, Range2), !.
d_prove([Atom | Atoms], Prog, DataPatch, Range1, Range2):-
    % Atom is a list of Constant/Variables
    Prog = ps(Ms, _, _, _),
    % get all possible clauses to prove
    my_findall([RuleName, MetaSub],
               (M = metasub(RuleName, MetaSub), element(M, Ms)),
               Rule_Mss),
    d_prove_all(Rule_Mss, Atom, Prog, DataPatch, Range1, Range3),
    %metarule(RuleName, MetaSub, (Atom:-Body), PreTest, Prog),
    %write('D: TRYING CLAUSE:'), nl, printprog([metasub(RuleName,MetaSub)]), 
    %call(PreTest),
    %write('D: Passed PreTest'), nl, 
    %d_prove(Body, Prog, DataPatch, Range1, Range3),
    d_prove(Atoms, Prog, DataPatch, Range3, Range2).

d_prove_all([], _, _, _, _, Range2, Range2):-
    Range2 \== [], !.
d_prove_all([[RuleName, MetaSub] | RMs], Atom, Prog,
            DataPatch, Range1, Range2, Temp_R):-
    metarule(RuleName, MetaSub, (Atom:-Body), PreTest, Prog),
    call(PreTest),
    d_prove(Body, Prog, DataPatch, Range1, Range3),
    union(Temp_R, Range3, Temp_R1),
    d_prove_all(RMs, Atom, Prog, DataPatch, Range1, Range2, Temp_R1).

d_prove_all(Rule_Mss, Atom, Prog, DataPatch, Range1, Range2):-
    d_prove_all(Rule_Mss, Atom, Prog, DataPatch, Range1, Range2, []).

%=========================================
% deductively prove one predicate
% @Atom: input Atom
% @Prog: input Program
% @DataPatch: input data
% @Range1: input data range
% @Range2: output data range (satisfied)
%=========================================
d_prove_pred_range([stat_classifier, _, _, C, _], _,
                        DataPatch, Range1, Range2):-
    % if the predicate is stat_classifier/4
    stat_classifier(DataPatch, Range1, C, Range2),
    Range2 \== [], !.
d_prove_pred_range([Pred, _, _, _], _, DataPatch, Range1, Range2):-
    callatom([Pred, DataPatch, Range1, Range2]),
    Range2 \== [], !.

%================================================================
% Recusively learning (sequencial coverage)
% if the coverage of previously learned program is not complete, 
% run another try on the rest data.
% @ FinalRange is the coverage of the final program
%=================================================================
recursive_learn(_, Prog1, DataPatch,
                Range1, Prog2, FinalRange, Temp_FR):-
    Prog1 = ps(_, _, Lim3, _),
    (Range1 == [];
     \+data_range_chk_positive(DataPatch, Range1);
     Lim3 == 0),
    sort(Temp_FR, FinalRange),
    Prog2 = Prog1, !.
recursive_learn(Pred/A, Prog1, DataPatch,
                Range1, Prog2, FinalRange, Temp_FR):-
    temp_vars(A, Vars),
    Atom_v = [Pred | Vars], % variable template for goal predicate
    learn_range([Atom_v], Prog1, DataPatch, Range1, Prog3, Range3),
    append(Temp_FR, Range3, Temp_FR1),
    list_complement(Range1, Range3, RangeC),
    recursive_learn(Pred/A, Prog3, DataPatch, RangeC,
                    Prog2, FinalRange, Temp_FR1).
recursive_learn(Pred/A, Prog1, DataPatch,
                Range1, Prog2, FinalRange):-
    recursive_learn(Pred/A, Prog1, DataPatch,
                Range1, Prog2, FinalRange, []).

%======================================================
% Learn predicate with predicate invetion handeling
% @Int: the interval of trying predicate invention
% @Pred/A: predicate name and arity
% @Prog0: initial program
% @DataPatch: input Data pointer
% @Range: initial data range
% @Prog2: output program
% ------------------------
% eval_range: an evaluation of range and data
%======================================================
learn_pred_invt(Int, Pred/A, Prog0, DataPatch, Range1, Prog2, EVal):-
    Prog0 = ps(Ms0, sig(Ps0, Cs0), Lim1, MetaRules),
    %write('Prog0: '), write(Prog0), nl,
    element(N, Int),
    peano(N, Lim1),
    write('N and Lim1: '), write(N), write(' '), write(Lim1), nl,
    name(Pred, PredChars), % char of label name
    N1 is N - 1,
    addnewpreds(PredChars, 0, N1, Ps0, Ps1),
    Prog1 = ps(Ms0, sig([Pred/A | Ps1], Cs0), Lim1, MetaRules),
    write('Program to be learned: '), write(Prog1), nl,
    %test_learn([Atom_v-Range], Prog1, DataPatch, Prog2).
    recursive_learn(Pred/A, Prog1, DataPatch, Range1, Prog2, FinalRange),
    write("Final Range: "), nl,
    write(FinalRange), nl,
    eval_range(DataPatch, FinalRange, EVal),
    write("AUC: "), write(EVal), nl,
    (EVal == 1.0, !, true
     ;
     true).

%==============================================================
% Learn a predicate with input data pointer, return hypothesis
% @Pred/A: Target predicate and arity
% @BK: background knowledge as program ps(_, _, _, _).
% @DataPatch: input Data pointer
% @Hyp: output hypothesis
%==============================================================
learn_single(Pred/A, BK, DataPatch, N_Clauses, Hyp):-
    interval(1, N_Clauses, I), % interval of learn new predicates
    get_data_size(DataPatch, Size), % get datapatch size (cpp lib)
    S is Size - 1,
    interval(0, S, Range1),
    %write(Range1), nl,
    my_findall([H, E],
               learn_pred_invt(I, Pred/A, BK, DataPatch, Range1, H, E),
               HEs
              ),
    find_best_hyp(HEs, Hyp).
    %learn_pred_invt(I, Pred/A, BK, DataPatch, Range1, Hyp).

%===================================================
% find_best_hyp(Hyps, Errs, Hyp).
% find best hypothesis according to error (AUC)
% @Hyps: list of hypotheses
% @Errs: list of errors (AUCs) corresponding to @Hyps
% @Hyp: returned best hypothesis
%===================================================
find_best_hyp([], Hyp, Hyp, _):-
    !.
find_best_hyp([HE | _], Hyp, _, _):-
    HE = [H, E],
    E == 1.0,
    Hyp = H, !.
find_best_hyp([HE | HEs], Hyp, _, Temp_E):-
    HE = [H, E],
    E >= Temp_E,
    Temp1 = H,
    Temp_E1 = E,
    find_best_hyp(HEs, Hyp, Temp1, Temp_E1), !.
find_best_hyp([HE | HEs], Hyp, Temp, Temp_E):-
    HE = [_, E],
    E < Temp_E,
    find_best_hyp(HEs, Hyp, Temp, Temp_E), !.
find_best_hyp(HEs, Hyp):-
    find_best_hyp(HEs, Hyp, [], -100).
%===============================================
% embed_bk(BK0, Hyp, BK1)
% Embed a hypothesis into background knowledge
% @BK0: Original background knowledge
% @Hyp: Newly learned hypothesis
% @BK1: Returned new background knowledge
%===============================================
embed_bk(BK0, Hyp, BK1):-
    BK0 = ps(Ms0, sig(Ps0, Cs0), _, MetaRules),
    Hyp = ps(Msh, sig(Psh, Csh), _, _),
    list_add_nodup(Msh, Ms0, Ms1),
    list_add_nodup(Psh, Ps0, Ps1),
    list_add_nodup(Csh, Cs0, Cs1),
    BK1 = ps(Ms1, sig(Ps1, Cs1), _, MetaRules).
    
% Findout wether a label is learned in background knowledge
% @BK: background knowledge
% @Label: name of label
/* learned(bk()):-


*/

%=================================
% Abduce program with metasub
% @MetaSub: metasub from metarule
% @Prog1: input program
% @Prog2: output program
%=================================
abduce(MetaSub, Prog, Prog):-
    ground(MetaSub), MetaSub, !.	% Ground call
abduce(MetaSub, ps(Ms, S, s(N), MRs), ps([MetaSub | Ms], S, N, MRs)):-
    MetaSub, !, ground(MetaSub).	% Capture constants
/*
abduce(MetaSub,Prog,Prog) :- Prog=ps(Ms,_,_,_),
			     element(MetaSub,Ms), !.		% Already in Program
*/
abduce(MetaSub, ps(Ms, S, s(N), Mss), ps([MetaSub | Ms], S, N, Mss)):-
    %write('INSTANCE CHECK'), nl, 
    %write(MetaSub), nl, 
    MetaSub = metasub(RuleName,[P/A | _]).

%================
% initialization
%================
init_ps(ps(BK, sig(Ps, Cs), _, Ms)):-
    init_bk(BK),
    init_metarules(Ms),
    init_preds(Ps),
    asserta_pred(Ps),
    init_consts(Cs).

init_preds(P):-
    primitives(P).

init_metarules(M):-
    metarules(M).

init_bk([]).

init_consts([]).  % Initial constants

%======================
% Meta rule definition
%======================
metarule(RuleName, MetaSub, Rule, PreTest, Program):-
        Program = ps(_, _, _, MetaRules),
        element(RuleName, MetaRules),
        metarule_(RuleName, MetaSub, Rule, PreTest, Program).

%==============================================
% Add arity "N" to every predicate "P" in "Ps"
%==============================================
add_arity(Ps, N, Rs):-
    add_arity(Ps, N, [], Rs).
add_arity([], _, Re, Re).
add_arity([P | Ps], N, Rs, Re):-
    append(Rs, [P/N], Rss),
    add_arity(Ps, N, Rss, Re).

%=========================
% Add range to predicates
%=========================
add_range(Ps, Range, Re):-
    add_range(Ps, Range, [], Re).
add_range([], _, Re, Re).
add_range([P | Ps], Range, Rs, Re):-
    append(Rs, [P-Range], Rss),
    add_range(Ps, Range, Rss, Re).

