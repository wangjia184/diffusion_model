/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { backend_util, RotateWithOffset, util } from '@tensorflow/tfjs-core';
export const rotateWithOffsetConfig = {
    kernelName: RotateWithOffset,
    backendName: 'cpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { image } = inputs;
        const { radians, fillValue, center } = attrs;
        const cpuBackend = backend;
        const output = util.getTypedArrayFromDType(image.dtype, util.sizeFromShape(image.shape));
        const [batch, imageHeight, imageWidth, numChannels] = image.shape;
        const [centerX, centerY] = backend_util.getImageCenter(center, imageHeight, imageWidth);
        const fullOpacityValue = 255;
        const sinFactor = Math.sin(radians);
        const cosFactor = Math.cos(radians);
        const imageVals = cpuBackend.data.get(image.dataId).values;
        for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
            const batchOffset = batchIdx * imageWidth * imageHeight * numChannels;
            for (let row = 0; row < imageHeight; row++) {
                const rowOffset = row * (imageWidth * numChannels);
                for (let col = 0; col < imageWidth; col++) {
                    const colOffset = col * numChannels;
                    for (let channel = 0; channel < numChannels; channel++) {
                        const coords = [batch, row, col, channel];
                        const x = coords[2];
                        const y = coords[1];
                        // coordX/coordY are the result of rotating and translating x/y.
                        let coordX = (x - centerX) * cosFactor - (y - centerY) * sinFactor;
                        let coordY = (x - centerX) * sinFactor + (y - centerY) * cosFactor;
                        coordX = Math.round(coordX + centerX);
                        coordY = Math.round(coordY + centerY);
                        let outputValue = fillValue;
                        if (typeof fillValue !== 'number') {
                            if (channel === 3) {
                                outputValue = fullOpacityValue;
                            }
                            else {
                                outputValue = fillValue[channel];
                            }
                        }
                        // If the coordinate position falls within the image boundaries...
                        if (coordX >= 0 && coordX < imageWidth && coordY >= 0 &&
                            coordY < imageHeight) {
                            // set the output to the image value at the coordinate position.
                            const rotatedRowOffset = coordY * (imageWidth * numChannels);
                            const rotatedColOffset = coordX * numChannels;
                            const imageIdx = batchOffset + rotatedRowOffset + rotatedColOffset + channel;
                            outputValue = imageVals[imageIdx];
                        }
                        const outIdx = batchOffset + rowOffset + colOffset + channel;
                        output[outIdx] = outputValue;
                    }
                }
            }
        }
        const dataId = cpuBackend.write(output, image.shape, image.dtype);
        return { dataId, shape: image.shape, dtype: image.dtype };
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUm90YXRlV2l0aE9mZnNldC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvUm90YXRlV2l0aE9mZnNldC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsWUFBWSxFQUFFLGdCQUFnQixFQUFpRCxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUkxSCxNQUFNLENBQUMsTUFBTSxzQkFBc0IsR0FBaUI7SUFDbEQsVUFBVSxFQUFFLGdCQUFnQjtJQUM1QixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsQ0FBQyxFQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUUsT0FBTyxFQUFDLEVBQUUsRUFBRTtRQUN2QyxNQUFNLEVBQUMsS0FBSyxFQUFDLEdBQUcsTUFBZ0MsQ0FBQztRQUNqRCxNQUFNLEVBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxNQUFNLEVBQUMsR0FDaEMsS0FBeUMsQ0FBQztRQUM1QyxNQUFNLFVBQVUsR0FBRyxPQUF5QixDQUFDO1FBRTdDLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxzQkFBc0IsQ0FDdEMsS0FBSyxDQUFDLEtBQXdCLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztRQUNyRSxNQUFNLENBQUMsS0FBSyxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUUsV0FBVyxDQUFDLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQztRQUVsRSxNQUFNLENBQUMsT0FBTyxFQUFFLE9BQU8sQ0FBQyxHQUNwQixZQUFZLENBQUMsY0FBYyxDQUFDLE1BQU0sRUFBRSxXQUFXLEVBQUUsVUFBVSxDQUFDLENBQUM7UUFDakUsTUFBTSxnQkFBZ0IsR0FBRyxHQUFHLENBQUM7UUFFN0IsTUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxP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QUFHLFVBQVUsQ0FBQyxLQUFLLENBQUMsTUFBTSxFQUFFLEtBQUssQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ2xFLE9BQU8sRUFBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLEtBQUssQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLEtBQUssQ0FBQyxLQUFLLEVBQUMsQ0FBQztJQUMxRCxDQUFDO0NBQ0YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIE51bWVyaWNEYXRhVHlwZSwgVHlwZWRBcnJheX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7YmFja2VuZF91dGlsLCBSb3RhdGVXaXRoT2Zmc2V0LCBSb3RhdGVXaXRoT2Zmc2V0QXR0cnMsIFJvdGF0ZVdpdGhPZmZzZXRJbnB1dHMsIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcblxuZXhwb3J0IGNvbnN0IHJvdGF0ZVdpdGhPZmZzZXRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogUm90YXRlV2l0aE9mZnNldCxcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiAoe2lucHV0cywgYXR0cnMsIGJhY2tlbmR9KSA9PiB7XG4gICAgY29uc3Qge2ltYWdlfSA9IGlucHV0cyBhcyBSb3RhdGVXaXRoT2Zmc2V0SW5wdXRzO1xuICAgIGNvbnN0IHtyYWRpYW5zLCBmaWxsVmFsdWUsIGNlbnRlcn0gPVxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBSb3RhdGVXaXRoT2Zmc2V0QXR0cnM7XG4gICAgY29uc3QgY3B1QmFja2VuZCA9IGJhY2tlbmQgYXMgTWF0aEJhY2tlbmRDUFU7XG5cbiAgICBjb25zdCBvdXRwdXQgPSB1dGlsLmdldFR5cGVkQXJyYXlGcm9tRFR5cGUoXG4gICAgICAgIGltYWdlLmR0eXBlIGFzIE51bWVyaWNEYXRhVHlwZSwgdXRpbC5zaXplRnJvbVNoYXBlKGltYWdlLnNoYXBlKSk7XG4gICAgY29uc3QgW2JhdGNoLCBpbWFnZUhlaWdodCwgaW1hZ2VXaWR0aCwgbnVtQ2hhbm5lbHNdID0gaW1hZ2Uuc2hhcGU7XG5cbiAgICBjb25zdCBbY2VudGVyWCwgY2VudGVyWV0gPVxuICAgICAgICBiYWNrZW5kX3V0aWwuZ2V0SW1hZ2VDZW50ZXIoY2VudGVyLCBpbWFnZUhlaWdodCwgaW1hZ2VXaWR0aCk7XG4gICAgY29uc3QgZnVsbE9wYWNpdHlWYWx1ZSA9IDI1NTtcblxuICAgIGNvbnN0IHNpbkZhY3RvciA9IE1hdGguc2luKHJhZGlhbnMpO1xuICAgIGNvbnN0IGNvc0ZhY3RvciA9IE1hdGguY29zKHJhZGlhbnMpO1xuICAgIGNvbnN0IGltYWdlVmFscyA9IGNwdUJhY2tlbmQuZGF0YS5nZXQoaW1hZ2UuZGF0YUlkKS52YWx1ZXMgYXMgVHlwZWRBcnJheTtcblxuICAgIGZvciAobGV0IGJhdGNoSWR4ID0gMDsgYmF0Y2hJZHggPCBiYXRjaDsgYmF0Y2hJZHgrKykge1xuICAgICAgY29uc3QgYmF0Y2hPZmZzZXQgPSBiYXRjaElkeCAqIGltYWdlV2lkdGggKiBpbWFnZUhlaWdodCAqIG51bUNoYW5uZWxzO1xuXG4gICAgICBmb3IgKGxldCByb3cgPSAwOyByb3cgPCBpbWFnZUhlaWdodDsgcm93KyspIHtcbiAgICAgICAgY29uc3Qgcm93T2Zmc2V0ID0gcm93ICogKGltYWdlV2lkdGggKiBudW1DaGFubmVscyk7XG5cbiAgICAgICAgZm9yIChsZXQgY29sID0gMDsgY29sIDwgaW1hZ2VXaWR0aDsgY29sKyspIHtcbiAgICAgICAgICBjb25zdCBjb2xPZmZzZXQgPSBjb2wgKiBudW1DaGFubmVscztcblxuICAgICAgICAgIGZvciAobGV0IGNoYW5uZWwgPSAwOyBjaGFubmVsIDwgbnVtQ2hhbm5lbHM7IGNoYW5uZWwrKykge1xuICAgICAgICAgICAgY29uc3QgY29vcmRzID0gW2JhdGNoLCByb3csIGNvbCwgY2hhbm5lbF07XG5cbiAgICAgICAgICAgIGNvbnN0IHggPSBjb29yZHNbMl07XG4gICAgICAgICAgICBjb25zdCB5ID0gY29vcmRzWzFdO1xuXG4gICAgICAgICAgICAvLyBjb29yZFgvY29vcmRZIGFyZSB0aGUgcmVzdWx0IG9mIHJvdGF0aW5nIGFuZCB0cmFuc2xhdGluZyB4L3kuXG4gICAgICAgICAgICBsZXQgY29vcmRYID0gKHggLSBjZW50ZXJYKSAqIGNvc0ZhY3RvciAtICh5IC0gY2VudGVyWSkgKiBzaW5GYWN0b3I7XG4gICAgICAgICAgICBsZXQgY29vcmRZID0gKHggLSBjZW50ZXJYKSAqIHNpbkZhY3RvciArICh5IC0gY2VudGVyWSkgKiBjb3NGYWN0b3I7XG4gICAgICAgICAgICBjb29yZFggPSBNYXRoLnJvdW5kKGNvb3JkWCArIGNlbnRlclgpO1xuICAgICAgICAgICAgY29vcmRZID0gTWF0aC5yb3VuZChjb29yZFkgKyBjZW50ZXJZKTtcblxuICAgICAgICAgICAgbGV0IG91dHB1dFZhbHVlID0gZmlsbFZhbHVlO1xuICAgICAgICAgICAgaWYgKHR5cGVvZiBmaWxsVmFsdWUgIT09ICdudW1iZXInKSB7XG4gICAgICAgICAgICAgIGlmIChjaGFubmVsID09PSAzKSB7XG4gICAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSBmdWxsT3BhY2l0eVZhbHVlO1xuICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIG91dHB1dFZhbHVlID0gZmlsbFZhbHVlW2NoYW5uZWxdO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIC8vIElmIHRoZSBjb29yZGluYXRlIHBvc2l0aW9uIGZhbGxzIHdpdGhpbiB0aGUgaW1hZ2UgYm91bmRhcmllcy4uLlxuICAgICAgICAgICAgaWYgKGNvb3JkWCA+PSAwICYmIGNvb3JkWCA8IGltYWdlV2lkdGggJiYgY29vcmRZID49IDAgJiZcbiAgICAgICAgICAgICAgICBjb29yZFkgPCBpbWFnZUhlaWdodCkge1xuICAgICAgICAgICAgICAvLyBzZXQgdGhlIG91dHB1dCB0byB0aGUgaW1hZ2UgdmFsdWUgYXQgdGhlIGNvb3JkaW5hdGUgcG9zaXRpb24uXG4gICAgICAgICAgICAgIGNvbnN0IHJvdGF0ZWRSb3dPZmZzZXQgPSBjb29yZFkgKiAoaW1hZ2VXaWR0aCAqIG51bUNoYW5uZWxzKTtcbiAgICAgICAgICAgICAgY29uc3Qgcm90YXRlZENvbE9mZnNldCA9IGNvb3JkWCAqIG51bUNoYW5uZWxzO1xuICAgICAgICAgICAgICBjb25zdCBpbWFnZUlkeCA9XG4gICAgICAgICAgICAgICAgICBiYXRjaE9mZnNldCArIHJvdGF0ZWRSb3dPZmZzZXQgKyByb3RhdGVkQ29sT2Zmc2V0ICsgY2hhbm5lbDtcbiAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSBpbWFnZVZhbHNbaW1hZ2VJZHhdO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBjb25zdCBvdXRJZHggPSBiYXRjaE9mZnNldCArIHJvd09mZnNldCArIGNvbE9mZnNldCArIGNoYW5uZWw7XG4gICAgICAgICAgICBvdXRwdXRbb3V0SWR4XSA9IG91dHB1dFZhbHVlIGFzIG51bWJlcjtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICBjb25zdCBkYXRhSWQgPSBjcHVCYWNrZW5kLndyaXRlKG91dHB1dCwgaW1hZ2Uuc2hhcGUsIGltYWdlLmR0eXBlKTtcbiAgICByZXR1cm4ge2RhdGFJZCwgc2hhcGU6IGltYWdlLnNoYXBlLCBkdHlwZTogaW1hZ2UuZHR5cGV9O1xuICB9XG59O1xuIl19