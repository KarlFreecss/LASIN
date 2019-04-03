%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learn statistical classifiers by abduction
%      AUTHOR: WANG-ZHOU DAI
% ========= Evaluate an hypothesis
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

%==============================================================
% Evaluate a predicate with input data pointer, Return results
% according to given Eval_method
% @Pred/A: Target predicate and arity
% @BK: background knowledge as program ps(_, _, _, _).
% @DataPatch: input Data pointer
% @Hyp: output hypothesis
%==============================================================

eval_pred_in_hyp(Pred/A, Prog, DataPatch, Eval_method, Val):-
    write("Pred: "), write(Pred/A), nl,
    write("Hyp: "), write(Prog), nl,
    temp_vars(A, Vars),
    Atom_v = [Pred | Vars], % variable template for goal predicate
    write("Atom: "), write(Atom_v), nl,
    get_data_size(DataPatch, Size),
    S is Size - 1,
    interval(0, S, Range1),
    % write("Range1: "), write(Range1), nl,
    prove_range([Atom_v], Prog, DataPatch, Range1, Range2),
    write("Range2: "), write(Range2), nl,
    callatom([Eval_method, DataPatch, Range2, Val]).

%==============================================================
% prove_range(Atom, DataPatch, Program, Range1, Range2).
% deductively evaluate @Program,
% input original @Range1
% output satisfied @Range2
%==============================================================
prove_range([], Prog, _, Range, Range).
prove_range([Atom | Atoms], Prog, DataPatch, Range1, Range2):-
    d_prove([Atom], Prog, DataPatch, Range1, Range3),
    prove_range(Atoms, Prog, DataPatch, Range3, Range2).

