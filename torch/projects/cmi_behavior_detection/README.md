This is my solution of [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/overview) Kaggle competition. The solution folder contains the entire code.

The solution contains two models: a soft label model and a hard label model. We average their predictions to compute the final predictions. Both models are largely identical except the training methodologies. The soft labels are generated using a cut-mix augmentation.
