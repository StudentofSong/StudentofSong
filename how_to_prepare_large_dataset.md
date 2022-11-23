# Processing, preparation and training of large datasets

## Background

The ways we used to process audio data on small data sets was：

1. Using kaldi/torchaudio to carry out data preprocessing to generate an `.ark/.scp` file, wherein the.ark file stores a binary file of the characteristics, the`.scp` file is an index of the `.ark` file, and a `text` file is used as a label corresponding to the audio;

2. Pack the data into a format that is easy for python to read. For specific code, refer to [code](https://github.com/maxwellzh/Transducer-dev/blob/e192070011b8e3ffa9ed818981e9321f12fe8117/cat/utils/pipeline/asr.py#L198).In this process, we will save the feature frame length information for the convenience of subsequent dynamic batching; It will also padding the label sequence (encoded as a number by the tokenizer) so that it can be saved in `numpy.ndarray` format and saved the index corresponding to the feature (like `.scp `file). The final saved file is the integration of the above several files.

In this process, the feature processing phase in step 1 spends most of time ,  and the step 2 takes very little time,  only a few minutes are needed to process 1000 hours of data (limited by hard disk IO).

When using data (model training), data loading is developed based on the `torch` standard `Dataset` (map-style) interface, with the following characteristics:

1.  After setting `shuffle = True` , data loading is **completely random**: Any two sentences can appear in the same mini-batch. Many works have shown that the randomness of sentences in mini-batch is beneficial to the performance of NN model

2.  Although we read data by index (rather than loading all the data from the hard disk into memory at once), benefited by the OS-level memory caching mechanism, after one round (one epoch) iteration. If the memory is large enough, all the data will be loaded into the memory without the upper user being aware of it.

From the above discussion, we have actually overlooked one situation: what if the memory isn't big enough or the data is too big? This question brings us to the shortcomings of the normal data loading approach that we will discuss:

If there is not enough memory to fit the entire dataset, the training will still work(this is because we are loading by index, if we load all the data into memory before the training starts, it will directly trigger the SIGKILL caused by OOM, and the process will be killed). But at the OS level, a different story emerges: when the system memory is full and the new data to be read is not in the memory, the OS will clean up some of the data in the memory to make room for the new data, The data loading (Hard Disk--> Memory) is very slow (compared to Memory--> CPU/GPU). Because of NN model training often requires multiple iterations, each round of this slow data loading will lead to a waste of time. Especially when `Data size>> Memory size`, almost equivalent to reading data directly on the hard disk, which has lower random read and write performance (comparison results of speed, sequential memory access >> random memory access > sequential disk access >> random disk acess). The perception of the end-user is that the time overhead of the training iterations increases almost exponentially as the training data increases, which is unacceptable for training on extremely large data sets.

## Protocol

It 's a given that we can't scale infinitely in hardware to match larger data sets (in practice, about 1200 hours of 80-FBank data can fill 256GB of memory), so reading data from the hard drive (rather than faster memory) is an unavoidable problem. But notice that the sequential read performance of the hard disk is far greater than the random read performance, we can start from characteristic 1, the data load to be modified.

Solutions provided by [webdataset](https://github.com/webdataset/webdataset)：

Reduce randomness of data loading. As mentioned earlier, fully sequential reads has some performance impacts, but we can trade-off between the two. Divide the whole datasets into multiple small files (partitioning is called sharding, and small files together are called ark list), each file contains several sentences (such as 2000), shuffle once at the ark list level, and shuffle again in each ark file, which not only retains certain randomness, but also reduces random access, and can significantly improve IO performance.

Based on `webdataset `, when dealing with large data sets(depending on memory size, typically greater than 1500 hours) , we will transform the data preparation process into :

1. Consistent with the previous mode 1, feature preprocessing;

2. Package the features and label (text format) of 2000 sentences into a file for processing. This process does not involve computation, but is primarily doing a lot of IO operations.

At present, in order to be compatible with the traditional mode, step 1 and step 2 are separated, and step 1 and step 2 may be considered to be combined in the future to further improve the efficiency of feature processing.

**NOTE:**
The big difference with the traditional way is that the label will be saved in text format. This is because we may change the tokenizer during actual use. If we save the label ID, we have to run step 2 again, which is very unworthy. Once the label text is saved, the tokenizer does the on-the-fly encoding directly when the data is loaded, with negligible overhead. 

In particular, when using partial tokenizer, you must handle the blank space in label, for example:

SentencePiece tokenizer using Chinese character modeling (tokenizer training without spaces), if the spaces in label are not removed during data preparation here, they will be mapped to <unk>, which will seriously affect the model performance. Therefore, for the Chinese datasets, it is better to remove the blank space in the label first, and then carry out data sharding; For tokenizers that are insensitive to some whitespace (such as the Jieba participle tokenizer), whitespace does not affect participles, so it doesn't matter

Later we will consider the audio features and labels separately, the text file processing is relatively easy, one-time full load into memory will not bring too much overhead.

## Interface design

### data preparation
The code to complete step 2 using the `webdataset` is available in [code](https://github.com/maxwellzh/Transducer-dev/blob/main/egs/wenetspeech/local/prep_wds.py#L16).Function interfaces are:

```python
# The number of sentences saved in each file, no need to modify if there is no special need
UTTS_PER_FILE = 2000

def pack_data(
        # The number of sentences saved in each file, no need to modify if there is no special need
        f_scps: Union[List[str], str],
        # text file，The first column is the sentence ID, which must match the ID in the .scp file, and multiple file lists are supported.
        f_labels: Union[List[str], str],
        # output folder
        d_out: str,
        # format of output file
        fmt: str = "data-%05d.tar",
        # Length grouping configuration, for example "10:2000" indicates that only sentences with a length of 10-2000 frames are reserved, and multiple groups can be used
        # like，["10:500", "500:800", "800:1000", "1000:1200"],files of different length groups will be saved in corresponding folders
        filter_group: List[str] = None):
    ...
```

### model training

Set `train:option:large_dataset = True` in the `hyper-p.json` file, and set `train:option:tokenizer=xxx` and `train:option:trset=xxx`. trset as the format of output file at the same time, for example:

```
# During data processing, d_out='./data', filter_group=['10:1000', '1000:2000']
# Then trset can be specified as
# 1. only use sentences of length 10-1000
trset='./data/10_1000/data-*.tar'
# 2. use sentences of length 10-2000
trset='./data/{10_1000,1000_2000}/data-*.tar'
# 3. code debug, only use 10x2000 sentences
trset='./data/10_1000/data-0000{0..9}.tar'
```

The underlying `webdataset` interface call is implemented in [code](https://github.com/maxwellzh/Transducer-dev/blob/main/cat/shared/manager.py#L82)

**NOTE:**  Since the development set data itself is `shuffle = False`, and the amount of data is generally small, the development of set data is still loaded in the traditional way.
## DDP

All the above discussions are based on the situation of single machine and single card. When DDP multi-card training or multi-machine and multi-card training is involved, this problem will become more complicated, for example:

```
trset="data-0{0..2}.tar"    # Contains three tar files with a total of 3x2000 sentences
# suppose there are two processes (two GPUs) using DDP training at this time
# do shuffle at the ark_list level, and assign the ark file to two processes after shuffle
gpu0: data-00.tar
gpu1: data-02.tar, data-01.tar
```

The problem with this is that the amount of data on the two processes is different, and DDP is a synchronous gradient update training. If it is trained directly, gpu1 will always wait for gpu0 to synchronize, and gpu0 has finished all data traversal and exited. The solution proposed by [wenet-e2e](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md#qa) for this problem is to use `model.join()`.

We use a simpler and more direct way, when a process traverses all the data, it directly forces all processes to end the current iteration round (epoch), which reduces the amount of data trained in 1 epoch, but because we will iterate more times and shuffle ark_list again each time, the impact of this part is relatively small

> wenetspeech-L (~10000 hours) contains about 15 million sentences, which are processed to about 7500.tar files, trained on 8 GPUs, 7500% 8 = 4, and 4x2000 sentences are discarded per round

**NOTE:**
The above example is just for ease of understanding. In practice, webdataset will do some duplication of ark files, so that the ark file hierarchy can be divided evenly. However, since those sentences in the dataset are not divided evenly by 2000, there will be one (or more) ark file with fewer sentences than the others, resulting in <2000 sentences be discarded each time

## reference

1. [webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch. (github.com)](https://github.com/webdataset/webdataset)
2. [wenet/UIO.md at main · wenet-e2e/wenet (github.com)](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md)
3. [Distributed Training with Uneven Inputs Using the Join Context Manager — PyTorch Tutorials 1.10.1+cu102 documentation](https://pytorch.org/tutorials/advanced/generic_join.html#how-does-join-work)

