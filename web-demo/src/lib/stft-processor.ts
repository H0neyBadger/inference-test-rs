import {fft, util as fftUtil} from "fft-js";

const WINDOW = 320;
const STRIDE = 160;
const POOLING = 6;
let BUFFER_SIZE = 1;
while (BUFFER_SIZE < WINDOW) {
  BUFFER_SIZE = BUFFER_SIZE<<1
}

class StftWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // one (1, 99, 257)
    // two (1, 99, 43, 1)
    // three (99, 43, 1)
    // four (99, 43, 1)
    // const sample_rate = 16000;
    // const buffer_size = 512;
    // const spectra_per_sec = sample_rate / buffer_size; // 31.25
    // const window = 320;
    // const stride = 160;
    // const frequency_bins = window / 2 +1 
    // const frames =  (sample_rate - window) / stride + 1
    // shape = (frames, frequency_bin)
    //
    // sample comes by 128 value frame
    this._buffer = new Float32Array(BUFFER_SIZE); // 128 * 3
    this._hamming_window = new Float32Array(BUFFER_SIZE);
    this._values = 0;
    this._index = 0;

    const arg = Math.PI * 2.0 / WINDOW;
    for (let i = 0; i < WINDOW; i++)
    {
      const float_value = 0.5 - (0.5 * Math.cos(arg * (i + 0.5)));
      this._hamming_window[i] = float_value;
    }
    for (let i = WINDOW; i < BUFFER_SIZE; i++)
    {
      // add padding
      this._hamming_window[i] = 0.0;
    }
  }

  _push(value) {
    this._buffer[this._index] = value;
    this._values +=1;
    this._index = (this._index+1) % BUFFER_SIZE;
  }

  _get_window() {
    let start = this._index - this._values;
    if (start < 0) {
      start = BUFFER_SIZE + start;
    }
    this._values -= STRIDE;

    // console.log(this._hamming_window.map((v, idx) => (start + idx) % BUFFER_SIZE))
    // let tmp = this._hamming_window.map((v, idx) => this._buffer[(start + idx) % BUFFER_SIZE])
    // console.log(tmp[0], tmp[STRIDE])
    return new Float32Array(
      this._hamming_window.map((v, idx) => this._buffer[(start + idx) % BUFFER_SIZE] * v)
    )
  }

  transform_fft(values) {
    // FIXME due to missing Nyquist value
    // the fft size is 256 vs 257 (including Nyquist)
    function* polling(values) {
      // reduce the size of the output by pooling with average and same padding
      for (let i = 0; i < values.length; i+= POOLING) {
        let pool = values.slice(i, i+POOLING)

        let value = pool.map((v) => v.magnitude).reduce((a, b) => a + b) / pool.length;
        // now take the log to give us reasonable values to feed into the network
        yield Math.log10(value + Number.EPSILON)
      }
    }
    return new Float32Array(polling(values))
  }

  _send(value) {
    this.port.postMessage({
      eventType: 'fft',
      value: value,
      output: this.transform_fft(value),
    });
  }

  process([[input]], [[output]], parameters) {
    if (input == undefined) {
      return false;
    }
    for (let i = 0; i < input.length; i++) {
      this._push(input[i])
      // console.log(input[i])
      if (this._values >= WINDOW) {
        // number of values to makes a window
        const window = this._get_window();
        const phasors = fft(window);
        const frequencies = fftUtil.fftFreq(phasors, 16000);
        const magnitudes = fftUtil.fftMag(phasors); 
        // Nyquist is missing !
        const value = frequencies.map(function (f, ix) {
            // squared magnitudes
            return {frequency: f, magnitude: magnitudes[ix]**2};
        });
        this._send(value)
      }
      // output[i] = input[i];
    }
    return true;
  }

}

console.log(`Sample rate ${sampleRate}`);

registerProcessor('stft-processor', StftWorkletProcessor);


