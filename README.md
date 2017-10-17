# Prediction angle of elbow based on sEMG
---
* sEMG sensors x 4
* Rotary encoder
* Tensorflow

input data = [sensor1, sensor2, sensor3, sensor4, elbow angle]
output data = [degree] # one_hot of 130 degree (0~129)

## layers

(DNN) * y
(LSTM * sequence_length) * x
