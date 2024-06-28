This repository contains an example how to train a pytorch model on the Nautilus Kubernetes cluster.

## Prerequisites
- Access to a kubernetes cluster, e.g., follow https://docs.nationalresearchplatform.org/userdocs/start/get-access/
- Access to a kubernetes namespace. Here, the namespace `rsekube` is used.
- Installed and configured `kubectl` utility: https://docs.nationalresearchplatform.org/userdocs/start/quickstart/

## The model
The model is defined in `train.py`. As example, we use the mnist CNN from pytorch: https://github.com/pytorch/examples/tree/main/mnist

## Run training
To submit the job to train the model, run on the (local) terminal:
```bash
kubectl apply -f kube_train.yaml
```
The trained model will appear on the exposed persistent volume: https://rsekube.nrp-nautilus.io
Note that this volume is publicly available.

Do not forget to delete the job ones it is completed:
```bash
kubectl delete -f kube_train.yaml
```

## Test the model
Install the requirements and run `python test.py`. Alternatively, run the notebook `test.ipynb`.
This can be done locally or on a remote jupyter server. The model will be donwnloaded from the presitent volume.
