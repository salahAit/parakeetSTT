import os
import sys
import shutil
import json
import ffmpeg  # pip install ffmpeg-python
import tqdm
from faster_whisper import WhisperModel

# ==========================================
# ⚡ إعدادات TURBO (السرعة القصوى - شامل الصيغ)
# ==========================================

# مسارات المجلدات
VIDEO_FOLDER = "videos"                # لم يتم تغيير المصدر
OUTPUT_FOLDER = "subtitles-turbo-json" 

MODEL_PATH = "large-v3-turbo"  

# إعدادات الأداء
# تم تثبيت اللغة على العربية لضمان دقة اللهجة
LANGUAGE = "ar"               
BATCH_SIZE = 8                
COMPUTE_TYPE = "float16"        

# التوجيه (محسّن للهجة الجزائرية وعدم خلط الفرنسية)
INITIAL_PROMPT = (
    "تفريغ نصي دقيق لمحاضرة باللهجة الجزائرية والعربية الفصحى. "
    "المتحدث لا يستخدم الفرنسية. يوجد فترات صمت وانتقال بين متحدثين."
)

# ==========================================

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. تحميل الموديل (محلياً) ---
print(f"\n>> 📂 جاري البحث عن الموديل في المجلد المحلي: {MODEL_PATH}...")

required_files = ["model.bin", "config.json", "tokenizer.json", "vocabulary.json"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_PATH, f))]

if missing_files:
    print(f"❌ خطأ: بعض ملفات الموديل مفقودة في المجلد {MODEL_PATH}:")
    print(f"   المفقود: {missing_files}")
    print("👉 يرجى استخدام أوامر wget لتحميلها أولاً.")
    sys.exit(1)

try:
    print(f">> 🖥️  جاري تهيئة الموديل على الـ GPU...")
    # تم إزالة BatchedInferencePipeline لأنه لا يدعم التحكم الدقيق في VAD parameters بنفس سهولة الـ transcribe العادي
    model = WhisperModel(
        MODEL_PATH, 
        device="cuda", 
        compute_type=COMPUTE_TYPE
    )
    print(">> ✅ الموديل جاهز للعمل (Offline Mode).\n")

except Exception as e:
    print(f"❌ خطأ في تحميل الموديل: {e}")
    sys.exit(1)


# --- دوال مساعدة (FFmpeg) ---

def extract_audio(video_path, audio_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16k")
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg Error.", file=sys.stderr)
        raise

def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe["format"]["duration"])
    except:
        return 0

def format_timestamp(seconds, separator=","):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"

# --- دوال حفظ الملفات ---

def write_srt(segments, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start, ",")
            end = format_timestamp(seg.end, ",")
            text = seg.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_vtt(segments, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start, ".")
            end = format_timestamp(seg.end, ".")
            text = seg.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_txt(segments, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")

# --- دوال JSON ---

def create_flat_word_list(segments):
    all_words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability
                })
    return all_words

def write_word_map_json(flat_word_list, output_file):
    word_map = {str(i): item["word"] for i, item in enumerate(flat_word_list, start=1)}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(word_map, f, indent=2, ensure_ascii=False)

def write_word_timestamps_json(flat_word_list, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(flat_word_list, f, indent=2, ensure_ascii=False)


# --- المحرك الرئيسي ---

def main():
    # الصيغ الشاملة (فيديو وصوت)
    supported_extensions = (
        '.mp4', '.mkv', '.mov', '.avi', '.webm', '.ts', '.flv', '.wmv', # فيديو
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus'                # صوت
    )
    
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(supported_extensions)]
    
    if not video_files:
        print(f"⚠️ المجلد '{VIDEO_FOLDER}' فارغ أو لا يحتوي على ملفات مدعومة.")
        print(f"الصيغ المدعومة الآن: {supported_extensions}")
        return

    print(f"🎬 وجدنا {len(video_files)} ملف. انطلاق التيربو (Offline)...\n")

    for index, video_file in enumerate(video_files, start=1):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        base_name = os.path.splitext(video_file)[0]
        current_output_dir = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(current_output_dir, exist_ok=True)
        audio_path = os.path.join(current_output_dir, f"{base_name}.wav")
        
        print(f"🔹 [{index}/{len(video_files)}] {video_file}")
        success = False

        try:
            extract_audio(video_path, audio_path)
            duration = get_audio_duration(audio_path)
            
            print(f"   ⚡ Turbo Transcribe ({duration/60:.2f} min)...")
            pbar = tqdm.tqdm(total=int(duration), unit="sec", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
            
            segments_list = []
            
            # --- الإعدادات المحسنة للبث المباشر والفراغات ---
            segments_generator, _ = model.transcribe(
                audio_path,
                language=LANGUAGE,
                
                # 1. إعدادات كشف الصمت (الأهم لمنع الهلوسة في الفراغات)
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=1000, # أي صمت فوق ثانية يتم تجاهله تماماً
                    speech_pad_ms=400,            # هوامش للأمان لعدم قص الكلام
                    threshold=0.5
                ),

                # 2. إعدادات التوليد
                beam_size=5,             # تقليل الـ Beam لتقليل محاولة "اختراع" نص في الضوضاء
                best_of=5,
                temperature=0.0,         # الحرارة صفر للدقة
                condition_on_previous_text=False, # هام جداً: يمنع تكرار الجملة السابقة في الفراغات
                initial_prompt=INITIAL_PROMPT,
                
                # 3. فلاتر إضافية للجودة
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0, 
                compression_ratio_threshold=2.4,

                word_timestamps=True
            )
            
            for segment in segments_generator:
                segments_list.append(segment)
                pbar.n = min(int(segment.end), int(duration))
                pbar.refresh()
                
                # طباعة ملونة بسيطة
                start_fmt = format_timestamp(segment.start, separator=":")
                pbar.write(f"\033[93m[{start_fmt}]\033[0m {segment.text.strip()}") 
            
            pbar.n = int(duration)
            pbar.refresh()
            pbar.close()

            # حفظ النتائج
            flat_words = create_flat_word_list(segments_list)
            
            files_map = {
                "srt": os.path.join(current_output_dir, f"{base_name}.srt"),
                "vtt": os.path.join(current_output_dir, f"{base_name}.vtt"),
                "txt": os.path.join(current_output_dir, f"{base_name}.txt"),
                "json_map": os.path.join(current_output_dir, f"{base_name}_word_map.json"),
                "json_time": os.path.join(current_output_dir, f"{base_name}_word_timestamps.json")
            }
            
            write_srt(segments_list, files_map["srt"])
            write_vtt(segments_list, files_map["vtt"])
            write_txt(segments_list, files_map["txt"])
            write_word_map_json(flat_words, files_map["json_map"])
            write_word_timestamps_json(flat_words, files_map["json_time"])
            
            print(f"   ✅ تم الحفظ في: {current_output_dir}")
            success = True

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"   🚨 خطأ: {e}")
        
        finally:
            print("-" * 50)

    print("\n🏁 انتهت مهمة التيربو!")

if __name__ == "__main__":
    main()
