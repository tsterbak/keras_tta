# keras_tta
Simple test time augmentation (TTA) for keras python library.

So far the wrapper flips the images horizontally and vertically and averages the predictions of all flipped images.

>The intuition behind this is that even if the test image is not too easy to make a prediction, the transformations change it such that the model has higher chances of capturing the target shape and predicting accordingly.


### Example:

```python
tta_model = TTA_ModelWrapper(model)

predictions = tta_model.predict(X_test)
```
