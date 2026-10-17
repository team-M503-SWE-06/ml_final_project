

## Contribution Report
MLE-AI-201 Machine Learning — Final Project, teamML-Team-08
Repository:https://github.com/team-M503-SWE-06/ml_final_project
Everything below can be checked against the Git history ofmainand the merged pull requests.
## Team
Member (Surname, Name)  GitHub handle  Main responsibilities
Aliyeva, NuranaNurana100Experiments: metrics module, tree and SVM sweeps,
benchmark vs scikit-learn
Garibli, Sanansanan-garibliData: repo skeleton, cleaning, features, frozen split, tier
label, EDA,run_all
Jamalov, HumbatCamalzadehDecision tree: from-scratch tree, its tests, permutation
importance, tree hyperparameter search
Nematzada, Farahf4r4hhError analysis and itsrun_allstep; final report and
slides; test-suite and full-pipeline verification runs
Shukurov, ElsenelsensukurovSVM: Pegasos SVM, random Fourier features, SVM
tests and analysis
Per-member summary
Aliyeva, Nurana —Wrote the NumPy metrics modulesrc/evaluate.pywith tests that cross-check
every metric against scikit-learn (tests/test_evaluate.py), together with the validation-only sweeps
src/experiments/tree_sweeps.pyandsrc/experiments/svm_sweeps.py(PR #4; the earlier PRs #2 and
#3 were folded into it). Builtsrc/experiments/benchmark.py(PR #5: 5-fold CV mean±std, timings, a
single test-set evaluation) and reworked it in PR #14 (fixed configs, optional SVM, tree-structure comparison
with scikit-learn, sklearn baselines insidePipelines, smearing-corrected AZN metrics). Gave 5 PR reviews.
Garibli, Sanan —Created the repo skeleton (first commit) andsrc/data_prep.pyin PR #1: cleaning of
the raw bina.az dump, feature construction, the frozen stratified train/val/test split and the premium/standard
tier label, plus the EDA figures insrc/eda.py. In PR #16 he madesrc/run_all.pyrun every experiment
end to end and fixed thenumpypin inrequirements.txt. Gave 17 PR reviews, the most on the team.
Jamalov, Humbat —Implementedsrc/decision_tree.pyfrom scratch in PR #6 (Gini/entropy/MSE,
vectorised split search, four stopping rules,predict_proba, feature importances, input validation, growth
with an explicit stack) and its test suitetests/test_decision_tree.pyin PR #7 (node-by-node match with
scikit-learn on fully determined examples, stopping rules, validation; 24 tests at merge). Added permutation
importance with one-hot/missing-flag groups insrc/importance.pywithtests/test_importance.py(PR
#8), and ran the 5-fold CV hyperparameter search whose tree configs are used inbenchmark.py(TREE_REG_-
PARAMS,TREE_CLS_PARAMS). Reviewed #5 and #10 (4 reviews in total)
Nematzada, Farah —Wrotesrc/error_analysis.pyin PR #18: test-set error analysis of the decision
trees by price band, by district and by distance to the tier threshold, and hooked it intosrc/run_all.py
(section 7b), so that the one command also reproduces the error tables and figure. Its tree settings are the final
ones frombenchmark.py, so it analyses the same trees as the results table. Wrote the report and finalized the
slides. Outside Git (not visible in the history), she ran the test suite (53 tests passed) and the fullpython -m
src.run_allpipeline on her own machine and checked that the error-analysis step reproduces the benchmark
numbers (test RMSE on log price 0.332, MAE 0.193).
Shukurov, Elsen —Implemented the Pegasos SVM insrc/svm.py(solver, per-epoch objective, margin and
support vectors, random Fourier features, mini-batches and iterate averaging) withtests/test_svm.pyand
the analysis scriptsrc/experiments/svm_analysis.py, merged as PR #15. The work was first split into PRs
#10–#13, which were closed in favour of #15. Gave 5 PR reviews.
Rough effort split
The approximate percentages were agreed by the team.
## 1

MemberApprox. %  Merged PRs  Non-merge commits on
main
Lines added  Reviews
Aliyeva, Nurana∼20%    #4, #5, #14   8 (+3, 2nd e-mail)∼6755
## Garibli, Sanan∼20%    #1, #165∼136317
## Jamalov, Humbat∼20%    #6, #7, #8   20∼7524
## Nematzada, Farah∼20%    #18, 19, 20    2∼1611
## Shukurov, Elsen∼20%    #1511∼7805
## 2