import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity


model_path = "Output/tf_model.h5"
loaded_model = tf.keras.models.load_model(model_path)

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.5,
        final_sparsity=0.9,
        begin_step=0,
        end_step=1000
    ),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

pruned_model = sparsity.prune_low_magnitude(loaded_model, **pruning_params)
pruned_model.save("Output/pruned_model.h5")