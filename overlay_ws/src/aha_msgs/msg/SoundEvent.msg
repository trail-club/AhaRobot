# One detected sound event. Producer: aha_perception (aha_audio_doorbell).
# header.stamp = end of the audio window in which the event was detected.

std_msgs/Header header
string          label       # "doorbell" | "knock"
float32         score       # post-processed score that reached the threshold
float32         threshold
string          model       # "<repo>@<revision[:7]>"
