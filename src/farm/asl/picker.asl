// BDI program of a strawberry-picking robot, in AgentSpeak(L) (Rao, 1996),
// run by the python-agentspeak interpreter (the language Jason uses).
//
// Beliefs revised every step from perception (see farm/bdi.py):
//   at_base, carrying, full, needs_charge, must_return, ripe_in_reach, ripe_in_sight,
//   has_path, returning, bored, mode(basic | coordinated), task(X, Y) per known crop,
//   and cautious if the robot uses the cautious commitment strategy.
// Mental state managed by the plans themselves:
//   committed(X, Y): the crop the robot intends to pick.
//
// Each step the body posts the goal !tick. The first applicable plan (in the order
// below) runs and performs exactly one action with .act, the robot's effector.

has_tasks :- task(_, _).

// ---- Desire: keep the robot working (one action per step) -------------------

+!tick : at_base & carrying       <- .act(unload).
+!tick : at_base & needs_charge   <- .act(charge).
+!tick : at_base & mode(basic)    <- .act(explore).
+!tick : at_base & has_tasks      <- !commit.
+!tick : at_base                  <- .act(wait_at_base).

+!tick : must_return              <- -committed(_, _); .act(return_to_base).
+!tick : ripe_in_reach & not full <- .act(pick).
+!tick : committed(X, Y) & has_path <- !pursue(X, Y).

+!tick : mode(basic) & ripe_in_sight      <- .act(approach_crop).
+!tick : mode(basic) & returning          <- .act(return_to_base).
+!tick : mode(basic) & carrying & bored   <- .act(return_to_base).
+!tick : mode(basic)                      <- .act(explore).

+!tick : has_tasks                <- !commit.
+!tick                            <- .act(return_to_base).

// ---- Intentions -------------------------------------------------------------

// Adopt the best known crop as the new intention.
+!commit : .best_task(X, Y) <- -committed(_, _); +committed(X, Y); .act(go_to_task, X, Y).
+!commit                    <- .act(return_to_base).

// Keep going to the committed crop. A *bold* robot never reconsiders on the way;
// a *cautious* one switches if a clearly better crop has become known.
+!pursue(X, Y) : cautious & .better_task(X, Y) <- !commit.
+!pursue(X, Y)                                 <- .act(move_to_crop).

// ---- Belief revision events -------------------------------------------------

// The intended crop is gone (picked by someone, or rotted): drop the intention.
-task(X, Y) : committed(X, Y) <- -committed(X, Y).
