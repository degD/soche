
const SAMPLE_RATE = 16_000;
const HOP_SIZE = 512;

const header = document.querySelector("h1");
const load_button = document.querySelector("button");
const audio_element = document.querySelector("audio");
const audio_source = document.querySelector("source");
const hidden_file_input = document.createElement("input");
hidden_file_input.type = "file";

let full_data;
let interval_id;

async function get_server_chords(file) {
  const response = await fetch("/get_chords", {
    method: "POST",
    body: file
  });
  const data = await response.json();

  full_data = []
  data.forEach(elm => {
    for (let i = 0; i < elm[0]; i++) {
      full_data.push(elm[1]);
    }
  });
}

hidden_file_input.addEventListener("change", () => {
  const file = hidden_file_input.files[0];
  const file_url = URL.createObjectURL(file);
  audio_source.src = file_url;

  get_server_chords(file).then(() => {
    audio_element.load();
  });
});

load_button.addEventListener("click", () => {
  hidden_file_input.click();
});

audio_element.addEventListener("playing", () => {
  interval_id = setInterval(() => {
    t = audio_element.currentTime;
    frame_id = Math.floor(t * SAMPLE_RATE / 512);
    header.textContent = full_data[frame_id];
  }, 10);
});
