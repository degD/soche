# SOCHE

SOCHE (SOng CHord Extraction) is a utility to extract correct chords from songs. There are some existing services available, but they are either bad, or expensive and closed source. I tried to make SOCHE both great, free and open source.

The goal is not to make a universal chord extraction utility. Existing research focus on complex chord structures and progressions. Instead, my target genre is pop. 

For a given audio, SOCHE tries to match it to the most fitting chord. First version of the utility tries to match the root note only. Second version tries to match major and minor qualities as well. Third version adds support for the remaining variants that repeat often in pop songs.

The target instrument for SOCHE is guitar. Therefore, guitar track separation may improve test results. 

Here is a diagram for SOCHE:

```
POP SONG  --->  SOCHE  --->  CHORDS
```

SOCHE will run a chord extraction algorithm repeatedly on a given pop song. The algorithm may be an AI as well. The algorithm runs for every 0.1s of the song, and outputs a chord guess. Subsequent outputs are merged if they are the same. Fluctations are detected and ignored. 

Some research suggest that having the song key note when running the chord extraction can improve test results. Therefore I may build a tool to find a key note of a pop song.

Target song format is mono channel, 44.1kHz, and 16-bit, WAV files. Songs are converted to this format prior to processing. Use the following command for conversion:

```
ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
```

## References


## License

MIT
