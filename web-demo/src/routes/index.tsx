import { useEffect, useState, useCallback, useRef } from 'react';
import { createFileRoute } from '@tanstack/react-router'
import {
  select,
  line,
  curveCardinal,
  scaleLinear,
  axisBottom,
  axisLeft,
} from "d3";


import { predict } from "inference-wasm";
import workletUrl from "../lib/stft-processor?worker&url";
// import marvinUrl from "../marvin/ab00c4b2_nohash_0.wav";

export const Route = createFileRoute('/')({ component: Home })

//chart component
const LineChart = ({data}) => {
  //refs
  const svgRef = useRef();

  //draws chart
  useEffect(() => {
    const svg = select(svgRef.current);

    //scales
    const xScale = scaleLinear()
      .domain([0, data.length - 1])
      .range([0, 400]);

    const yScale = scaleLinear().domain([0, 100]).range([100, 0]);

    //axes
    const xAxis = axisBottom(xScale).ticks(data.length);
    svg.select(".x-axis").style("transform", "translateY(100px)").call(xAxis);

    const yAxis = axisLeft(yScale);
    svg.select(".y-axis").style("transform", "translateX(0px)").call(yAxis);

    //line generator
    const myLine = line()
      .x((d, i) => xScale(i))
      .y((d) => yScale(d.y))
      .curve(curveCardinal);

    //drawing the line
    svg
      .selectAll(".line")
      .data([data])
      .join("path")
      .attr("class", "line")
      .attr("d", myLine)
      .attr("fill", "none")
      .attr("stroke", "#00bfa6");
  }, [data]);

  return (
      <svg ref={svgRef}/>
  );
};



const getMedia = async () => {
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: {sampleRate: 16000, channelCount: 1},
      video: false,
    })
  } catch (err) {
    console.log('Error:', err)
  }
}

const useStft = ({onFftUpdate, onStftUpdate}) => {
  const [running, setRunning] = useState(false)
  const spectrogram = useRef<Foat32Array[][]>([[]])
  
  const refresh = useCallback(({value, output}: {value: Float32Array[], output: Float32Array[]}) => {
    let stft = []
    onFftUpdate(Array.from(value).map((v, idx) =>({x:v.frequency, y:v.magnitude})))
    if (spectrogram.current.length >= 99) {
      stft = [...spectrogram.current.slice(1), output]
    } else {
      stft = [...spectrogram.current, output]
    }
    spectrogram.current = stft
    if (stft.length >= 99) {
      // make prediction
      const result = predict(spectrogram.current)
      if(result > 0.997) {
        console.log(`>>>>>>> prediction result ${result}, ${spectrogram.current}`)
      }
    }

  }, [spectrogram, predict])

  useEffect(() => {
    async function setupAudio() {
      const stream = await getMedia()

      const audioContext = new AudioContext({sampleRate: 16000});

      /*
      const source = audioContext.createBufferSource();
      const audioBuffer = await fetch(marvinUrl)
        .then(res => res.arrayBuffer())
        .then(ArrayBuffer => audioContext.decodeAudioData(ArrayBuffer));
      source.buffer = audioBuffer;
      */

      const source = audioContext.createMediaStreamSource(stream);
      await audioContext.audioWorklet.addModule(new URL("../lib/stft-processor.ts", import.meta.url))
      const stftNode = new AudioWorkletNode(
        audioContext,
        "stft-processor",
      );
      stftNode.port.onmessage = (e) => {
        if (e.data.eventType === 'fft') {
          // process pcm data
          refresh(e.data);
        }
      };
      // stftNode.connect(audioContext.destination);
      source.connect(stftNode).connect(audioContext.destination);
      // here
      // source.start()

    }
    setupAudio();
  }, [])

  return [running]
}


function Home() {
  const [fftData, setFftData] = useState([])
  const [running] = useStft({onFftUpdate:setFftData});

  return (
    <div className="p-8">
      <h1 className="text-4xl font-bold">Welcome to stft inference-test</h1>
      <p className="mt-4 text-lg">
        Edit <code>src/routes/index.tsx</code> to get started.
      </p>
        {/* <LineChart data={fftData}/> */}
    </div>
  )
}

