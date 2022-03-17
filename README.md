# Serving movie recommendation model with ray serve

First preprocess data
```shell
$ python preprocess.py 
```

Start ray processes
```shell
$ ray start --head
```

Run service
```shell
$ python recommender.py
```

If you want to stop the service
```shell
$ ray stop
```

