# NLP Suicide

What you'll find in this folder:

1. `models/` contains the pickled model filds
2. `output/` contains the output images


### Environment

The following are the commands I use to get the Python environment set up and working. 

```
cd nlp_suicide
module load python/3.6-anaconda-4.4
source activate myenv
jupyter notebook --port 9998
```

Make sure that when you SSH into CORI, you are mapping remote port 9998 to your local port 9998:

```
ssh -L 9998:localhost:9998 -l <username> cori.nersc.gov
```
