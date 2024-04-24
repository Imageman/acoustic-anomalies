# acoustic-anomalies
Proof of concept. Capture sound from a microphone, build a reference profile of noise and/or sounds and then search for anomalies (difference from the reference).

The first phase after launch is to build a profile:\
![build-profile.png](help%2Fbuild-profile.png)

After collecting the statistics, an acoustic profile (model) is built.\
![profile-completed.png](help%2Fprofile-completed.png)

In anomaly tracking mode, CPU utilization is about 1% (Intel i5).
When an anomaly is detected, a message is displayed and an mp3 file and a service mfcc file (not currently used) are written to disk.\
![anomaly_screenshot.png](help%2Fanomaly_screenshot.png)

An example of what the recordings look like in the audio editor.\
![sample1.png](help%2Fsample1.png)
![sample2.png](help%2Fsample2.png)
