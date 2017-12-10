# HOTS_QM_Predict
Simple Neural Network for Predicting HOTS Quickmatch results

Use of this model is pretty simple, to get started just reference the example notebook I've uploaded.

The model produces a single number representing a confidence of victory, so if you're trying to determine whether your comp is good, higher numbers are better.
Another thing to note is that this model is not necessarily accurate when prediction for any of the draft queue types, but it will be more right/wrong in some places than in others. On example is that a team of all assassins in quickmatch may have a ~50% chance to win because they will be paired against a similar composition, but that same team in any draft queue would almost certainly be decimated by any compentant team who picks a healthy balance of the classes. As such, if you do attempt to use this model in your draft queues, keep in mind that it will only have some semblance of accuracy in cases where your composition roughly matches up with that of your opponent. 
I plan to develope a similar model designed for draft queues.

If you have any questions or suggestions don't hesitate to post or message me on Reddit (my reddit ID is Xenoproboscizoid).
