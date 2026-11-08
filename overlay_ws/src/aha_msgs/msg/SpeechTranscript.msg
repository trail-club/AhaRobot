# One recognized utterance. Producer: aha_perception (aha_audio_asr).
# header.stamp = time the utterance ended (last voiced audio frame).

std_msgs/Header header
string          text
string          language         # "ja" | "en" (also "ja" with ja-qwen; see model)
string          model            # model that produced the text, "<repo>@<revision[:7]>"
float32         audio_duration   # seconds of speech in the segment
float32         latency          # seconds from end of speech to text available
