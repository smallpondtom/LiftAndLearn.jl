# Lift And Learn

Lift and Learn is a model reduction that allows us to deal with nonlinear PDEs which are not particularly in the form of a Polynomial system. This is made possible using the _lifting_ method which introduces an auxiliary variable into the system and lifts the system into a polynomial form. This is similar to Koopman theory. However, lifting is a direct transformation, whereas Koopman theory is an approximation.

After lifting the system, the Operator Inference scheme can be applied easily to discovery the reduced operators from data.

For further details on lifting, please see [lifting](../Lift.md).