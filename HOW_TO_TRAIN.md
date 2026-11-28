<h1>Step-By-Step Training Instructions</h1>
1) Download the dataset.<br>
2) Run
pip install -r requirements-lock-windows-py313.txt
<br>
    - This is to install neuraloperator, pytorch, etc.<br>
3) There are 3 important files:
    <br>-   scripts/train_numerical_fnn_with_validity.py
    <br>-   scripts/train_goes_fnn_with_validity.py
    <br>-   scripts/train_goes_overlaid_numerical.py
    <br>-   scripts/inspect_processed_data.py.



inspect_processed_data.py is where all of the loading and processing action really is!

4) Invoke numerical model training using the command
```
python ./scripts/train_numerical_fnn_with_validity.py [left train sampling bound] [right train sampling bound] [left test sampling bound] [right test sampling bound] [checkpoint saving directory] [epochs]
```

Parameters are as such:
<ul>
<li> [left train sampling bound] </li>
<li>[right train sampling bound] - What range of samples will be used for training a model </l1>
<li>[left test sampling bound] </li>
<li>[right test sampling bound] - What range of samples will be used on the test data </li>
<li> [checkpoint saving path] - The path/filename to which a model will be saved </li> 
<li>[epochs] - How many epochs the model will be trained for </li>
</ul>

i.e.
```
python ./scripts/train_numerical_fnn_with_validity.py 0 80 80 100 ./models/mymodel2 20 > out.json
```

You can invoke GOES model training similarly
```
python ./scripts/train_goes_fnn_with_validity.py [left train sampling bound] [right train sampling bound] [left test sampling bound] [right test sampling bound] [checkpoint saving directory] [epochs]
```

5) Invoke GOES + numerical overlay training using
```
python ./scripts/train_goes_overlaid_model.py [numerical_model_checkpoint_path] [left train sampling bound] [right train sampling bound] [left test sampling bound] [right test sampling bound] [checkpoint saving directory] [epochs]
```
Only one additional parameter
<ul>
<li> [numerical_model_checkpoint_path] - The path/filename where the numerical model is located </li> 
</ul>

i.e.

```
python ./scripts/train_goes_overlaid_numerical.py ./models/mymodel2 300 375 375 400 ./models/overlaid_model 20
```
The numerical model must be trained first before using the overlay model!

<h1>Possible Points of Error</h1>
<h2>Loss function calculation in MaskedMSE</h2>
<ul>
    <li>Currently loss is calculated with a custom class MaskedMeanSquaredError, which is a copy of neuralop.MeanSquaredError that takes into account interleaved masks in the channel axis.
    </l1>
    <li> This loss function may not be sufficient for our purposes - maybe use a different lossf
    </li>
    <li> The use of the validity mask may be incorrect - it is taken out of the predict/truth y-tensors when calculating MSE, and it is primarily used so that its sum (how many 1s there are) can divide each MSE
    </li>
</ul>
<h2>Validity mask usage</h2>
<ul>
    <li>The validity mask is provided to the model directly, it is folded directly into the input channel. The model must "learn" the use of the validity mask
    </l1>
    <li> The use of the validity mask must not be normalized when fed into both input and output tensors
    </li>
</ul>
<h2>Pre-Training Data Preprocessing</h2>
<ul>
    <li>Normalization must be applied to the input and output tensors AND it must not take into account or normalize validity masks
    <li> Need to take a look at provide_normalizer() - it may be taking into accoun the validity mask for normalization
</ul>
<h2>Model Configuration</h2>
<ul>
    <li>Current optimizer is AdamW; Scheduler is CosineAnnealingLR (i don't really know what those are)</li>
    <li>We are basically running the default settings on FNO</li>
</ul>