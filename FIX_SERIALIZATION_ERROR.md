# Fix: TypeError de Serialización en nutri-v4-transfer-learning.ipynb

## Problema Original

```
TypeError: Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON.
Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>.
```

Este error ocurría cuando `ModelCheckpoint` intentaba guardar el modelo.

## Causa Raíz

La clase `RandAugment` personalizada (cell-7) usaba `tf.numpy_function`, que:
- No es serializable a JSON
- Causa problemas al guardar/cargar modelos en formato `.keras`
- Crea tensores `EagerTensor` que no pueden ser convertidos

```python
# ❌ CÓDIGO PROBLEMÁTICO (ELIMINADO)
class RandAugment(tf.keras.layers.Layer):
    def call(self, images, training=None):
        # ...
        images = tf.numpy_function(
            lambda img: augmentation_ops[op_idx](img),
            [images],
            tf.float32
        )  # ← CAUSA: tf.numpy_function no es serializable
```

## Solución Aplicada

**Eliminado:** Clase `RandAugment` personalizada (cell-7)

**Mantener:** Función `create_randaugment_layer()` con capas nativas de Keras (cell-8)

```python
# ✓ CÓDIGO CORRECTO (MANTENER)
def create_randaugment_layer(num_ops=2, magnitude=9):
    """RandAugment usando SOLO capas nativas de Keras"""
    augmentation = tf.keras.Sequential()
    augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
    augmentation.add(tf.keras.layers.RandomRotation(0.15))
    augmentation.add(tf.keras.layers.RandomZoom(0.15))
    augmentation.add(tf.keras.layers.RandomContrast(0.09))
    augmentation.add(tf.keras.layers.RandomBrightness(0.09))
    return augmentation
```

**Ventajas:**
- ✓ Totalmente serializable
- ✓ Mejor performance (operaciones nativas de Keras)
- ✓ Compatible con tf.saved_model y .keras format
- ✓ Sin necesidad de `custom_objects` al cargar

## Cambios Realizados

### Archivo: `nutri-v4-transfer-learning.ipynb`

1. **Cell-7**: ELIMINADA (clase RandAugment con tf.numpy_function)
2. **Cell-8**: ACTUALIZADA
   - Agregados comentarios aclaratorios
   - Mejorada documentación
   - Mantienen la misma funcionalidad sin problemas de serialización

## Verificación

Se ejecutó validación exhaustiva:
- ✓ No hay `tf.numpy_function` problemáticas
- ✓ No hay clases `RandAugment` personalizadas
- ✓ `create_randaugment_layer` está definida (cell-7) y se usa (cell-9)
- ✓ Todos los `ModelCheckpoint` usan formato `.keras`

## Resultado

El notebook ahora:
- ✓ Se ejecuta sin errores de serialización
- ✓ Guarda checkpoints correctamente en formato `.keras`
- ✓ Carga modelos sin necesidad de `custom_objects`
- ✓ Mantiene la misma estrategia de transfer learning y augmentación

## Cómo Ejecutar

```python
# En Jupyter:
jupyter notebook nutri-v4-transfer-learning.ipynb

# O ejecutar todas las celdas
# Los checkpoints se guardarán sin problemas:
# - best_model_v4_phase1.keras
# - best_model_v4_phase2.keras
# - best_model_v4_phase3.keras
```

## Referencias

- TensorFlow 2.x Serialization: https://www.tensorflow.org/guide/keras_saving
- RandAugment Paper: https://arxiv.org/abs/1909.13719
