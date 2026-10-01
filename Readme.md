ANN Loan Approval Prediction
============================

A neural network based binary classification system that predicts whether a
loan application will be approved or rejected. Built using TensorFlow/Keras
on tabular data, achieving 90.1% test accuracy.

-------------------------------------------------------------------------------
OVERVIEW
-------------------------------------------------------------------------------

Problem Type      : Supervised Learning - Binary Classification
Dataset           : ANN_Loan_Approval_Dataset.csv
Framework         : TensorFlow 2.21 / Keras
Final Accuracy    : 90.1%
Total Parameters  : ~4,000

The model is trained on historical loan application data to learn patterns
that separate approved applications from rejected ones. Once trained, it can
score new applications instantly without manual review.

-------------------------------------------------------------------------------
DATASET
-------------------------------------------------------------------------------

Total Records                : 5,000
Raw Columns                  : 17
Features After Encoding      : 21
Target Column                : Loan_Approved
Class Distribution           : 3,417 approved (1) vs 1,583 rejected (0)
Class Ratio                  : 2.2 : 1 (imbalanced)

Categorical Columns
    Education, Employment_Type, Marital_Status, Property_Area, Existing_Loan

Numerical Columns
    Age, Dependents, Experience_Years, Monthly_Income, Credit_Score,
    Loan_Amount, Loan_Term_Months, EMI, Savings, Debt_to_Income_Ratio

Dropped Column
    Application_ID (no predictive value)

-------------------------------------------------------------------------------
PREPROCESSING PIPELINE
-------------------------------------------------------------------------------

Order of operations is critical. Skipping any step will degrade model
performance or introduce data leakage.

Step 1: Encode Categorical Variables
    Method  : pd.get_dummies(X, drop_first=True)
    Reason  : Neural networks cannot process text directly. drop_first=True
              removes one dummy column per category to avoid perfect
              multicollinearity (dummy variable trap).

Step 2: Train-Test Split
    Method  : train_test_split(X, y, test_size=0.2, random_state=42,
                                stratify=y)
    Reason  : 80/20 split gives enough data for training while keeping
              a representative test set. stratify=y preserves the 2.2:1
              class ratio in both splits.

Step 3: Feature Scaling (fit on training data only)
    Method  : StandardScaler().fit_transform(X_train)
    Reason  : The network uses gradient descent. Features on very different
              scales (Age around 60 vs Loan_Amount around 1,000,000) cause
              slow convergence and unstable weight updates. StandardScaler
              normalizes each feature to mean=0, std=1.

Step 4: Transform Test Data
    Method  : scaler.transform(X_test)
    Reason  : The scaler must be fit only on training data. Fitting on the
              combined dataset leaks test statistics into training, which
              inflates test performance.

-------------------------------------------------------------------------------
MODEL ARCHITECTURE
-------------------------------------------------------------------------------

Layer Configuration

    Input Layer          : 21 features
    Hidden Layer 1       : Dense(64, activation='relu')
    Dropout              : 0.3
    Hidden Layer 2       : Dense(32, activation='relu')
    Hidden Layer 3       : Dense(16, activation='relu')
    Output Layer         : Dense(1, activation='sigmoid')

Design Rationale

    ReLU in hidden layers
        Fast to compute. Does not saturate for positive inputs, so it avoids
        the vanishing gradient problem that affects sigmoid and tanh in deep
        networks.

    Sigmoid in output layer
        Compresses the output to the range [0, 1], which we interpret as the
        probability of class 1 (loan approved). For multi-class problems,
        softmax would be used instead.

    Dropout(0.3)
        Randomly deactivates 30% of neurons during each training step. This
        forces the network to learn redundant representations and significantly
        reduces overfitting. Dropout is automatically disabled during
        inference.

    Funnel structure (64 -> 32 -> 16 -> 1)
        Each successive layer compresses the representation, forcing the
        network to extract progressively higher-level abstractions.

    Parameter count (~4,000)
        Proportional to dataset size. A larger network on 4,000 training
        samples would overfit.

-------------------------------------------------------------------------------
COMPILATION AND TRAINING
-------------------------------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

Parameter Reference

    optimizer = 'adam'
        Adaptive learning rate per parameter. Combines momentum with RMSProp.
        Converges faster than plain SGD and rarely needs tuning.

    loss = 'binary_crossentropy'
        Standard loss for two-class problems. Penalizes confident wrong
        predictions heavily, which produces well-calibrated probabilities.

    metrics = ['accuracy']
        Fraction of correct predictions, tracked per epoch.

    epochs = 30
        Number of full passes over the training data.

    batch_size = 32
        Number of samples processed before each weight update. Smaller batch
        sizes produce noisier gradients but often generalize better.

    validation_split = 0.2
        Holds out 20% of training data to monitor overfitting at each epoch.
        This data is not used to update weights.

Training Progression

    Epoch 1   : train acc 0.7497  val acc 0.8462
    Epoch 10  : train acc 0.8847  val acc 0.8925
    Epoch 20  : train acc 0.8963  val acc 0.8888
    Epoch 30  : train acc 0.9128  val acc 0.8925

    Training and validation curves track closely, indicating the model is
    not overfitting.

-------------------------------------------------------------------------------
RESULTS
-------------------------------------------------------------------------------

Test Set Performance

    Test Accuracy    : 0.9010  (90.1%)
    Test Loss        : 0.2315

Confusion Matrix

                        Predicted 0     Predicted 1
    Actual 0                 249              68
    Actual 1                  31             652

    True Negatives   : 249   (correctly rejected)
    True Positives   : 652   (correctly approved)
    False Positives  :  68   (rejected loans predicted as approved)
    False Negatives  :  31   (approved loans predicted as rejected)

    The 68 false positives represent financial risk - the model approved
    applications that were historically rejected. The 31 false negatives
    represent lost business - good loans the model incorrectly rejected.

Classification Report

    Class               Precision   Recall   F1-Score   Support
    0 - Rejected            0.89      0.79      0.83        317
    1 - Approved            0.91      0.95      0.93        683
    Weighted Average        0.90      0.90      0.90       1000

Observation

    The model performs better on class 1 because it has more than twice
    the training samples. Class 0 recall of 0.79 is the weakest metric -
    the model misses approximately 21% of rejections. This is a direct
    consequence of the class imbalance and can be addressed with class
    weights or SMOTE.

-------------------------------------------------------------------------------
SINGLE PREDICTION
-------------------------------------------------------------------------------

# Take one applicant from the test set
sample = X_test.iloc[[0]]

# Apply the exact same scaler that was fit on training data
sample_scaled = scaler.transform(sample)

# Get probability of approval
probability = model.predict(sample_scaled)[0][0]

# Apply threshold
if probability > 0.5:
    prediction = 'Approved'
else:
    prediction = 'Rejected'

print('Probability:', probability)
print('Prediction :', prediction)

Sample Output

    Probability: 0.99987
    Prediction : Approved

-------------------------------------------------------------------------------
KEY CONCEPTS
-------------------------------------------------------------------------------

Why Sigmoid in the Output Layer

    Sigmoid maps any real number to the interval (0, 1). Since our target
    is binary, the output can be directly interpreted as P(approved).
    For multiclass problems, softmax is used to produce a probability
    distribution across classes.

Why Binary Crossentropy

    Measures the divergence between predicted probability and the true
    label. For a correct confident prediction the loss is near zero. For
    a confidently wrong prediction the loss becomes very large, which
    drives the optimizer to correct such errors. Mean squared error
    converges more slowly for probability outputs and is generally not
    recommended for classification.

Adam vs SGD

    Adam
        Maintains a separate adaptive learning rate for every parameter.
        Combines momentum with RMSProp. Converges quickly and requires
        little hyperparameter tuning. Good default for most problems.

    SGD
        Uses a single global learning rate. Slower to converge and often
        needs learning rate schedules or momentum. Can sometimes find
        better minima than Adam when tuned carefully.

Dropout Behavior

    During training, Dropout randomly zeros a fraction of neurons on each
    forward pass. During inference (predict / evaluate), Dropout is
    automatically disabled and all neurons are active. No manual switching
    is required.

-------------------------------------------------------------------------------
COMMON PITFALLS
-------------------------------------------------------------------------------

Data Leakage
    Fitting the scaler on the full dataset before splitting lets test
    statistics influence training. Always fit on training data only.

Class Imbalance
    90% accuracy looks strong, but class 0 recall is only 0.79. For
    imbalanced problems, accuracy alone is misleading. Monitor precision,
    recall, and F1 for each class.

Missing Validation
    Without validation_split (or a separate validation set), overfitting
    cannot be detected until test evaluation, by which time the training
    run is complete.

Fixed Threshold
    The default 0.5 threshold may not be optimal. If false positives are
    costlier than false negatives (or vice versa), adjust the threshold
    accordingly.

Oversized Network
    Adding more layers and neurons does not always help. On a 4,000-sample
    training set, a very deep network will overfit. Match model capacity
    to data size.

-------------------------------------------------------------------------------
PRACTICE EXPERIMENTS
-------------------------------------------------------------------------------

    Increase epochs from 30 to 50 and compare accuracy curves.

    Change Dropout from 0.3 to 0.5 and observe the effect on validation
    loss.

    Scale up the architecture from (64, 32, 16) to (128, 64, 32).

    Replace Adam with SGD (learning_rate=0.01) and compare convergence.

    Add EarlyStopping(patience=5, monitor='val_loss') to stop training
    automatically when validation loss stops improving.

    Handle class imbalance by passing class_weight={0: 2.2, 1: 1.0} to
    model.fit, or by applying SMOTE to the training set.

    Tune the decision threshold. Try 0.3, 0.4, 0.5 and compare precision,
    recall, and F1 for both classes.

-------------------------------------------------------------------------------
WORKFLOW SUMMARY
-------------------------------------------------------------------------------

    Preprocess  ->  Split  ->  Scale  ->  Build  ->  Compile  ->  Fit
                                                                    |
                                                                    v
                                    Predict  <-  Evaluate  <-  Trained Model

-------------------------------------------------------------------------------
TECH STACK
-------------------------------------------------------------------------------

    Python 3.13
    pandas, numpy        - data manipulation
    scikit-learn         - train/test split, StandardScaler, metrics
    TensorFlow 2.21      - neural network framework
    Keras                - high-level model API
    matplotlib           - accuracy and loss visualization