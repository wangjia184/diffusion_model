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
import { ResizeBilinear, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
export function resizeBilinear(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    assertNotComplex(images, 'resizeBilinear');
    const imagesStrides = util.computeStrides(images.shape);
    const [newHeight, newWidth] = size;
    const [batch, oldHeight, oldWidth, numChannels] = images.shape;
    const xValues = backend.data.get(images.dataId).values;
    const result = new Float32Array(util.sizeFromShape([batch, newHeight, newWidth, numChannels]));
    const effectiveInputSize = [
        (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
        (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];
    const effectiveOutputSize = [
        (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
        (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];
    let outputIdx = 0;
    const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
    const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
    for (let b = 0; b < batch; b++) {
        for (let r = 0; r < newHeight; r++) {
            let sourceFracRow;
            if (halfPixelCenters) {
                sourceFracRow = effectiveRowSizeRatio * (r + 0.5) - 0.5;
            }
            else {
                sourceFracRow = effectiveRowSizeRatio * r;
            }
            const sourceRowFloor = Math.max(0, Math.floor(sourceFracRow));
            const rowFrac = sourceFracRow - sourceRowFloor;
            const sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
            const topRowOffset = b * imagesStrides[0] + sourceRowFloor * imagesStrides[1];
            const botRowOffset = b * imagesStrides[0] + sourceRowCeil * imagesStrides[1];
            for (let c = 0; c < newWidth; c++) {
                let sourceFracCol;
                if (halfPixelCenters) {
                    sourceFracCol = effectiveColSizeRatio * (c + 0.5) - 0.5;
                }
                else {
                    sourceFracCol = effectiveColSizeRatio * c;
                }
                const sourceColFloor = Math.max(0, Math.floor(sourceFracCol));
                const colFrac = sourceFracCol - sourceColFloor;
                const sourceColCeil = Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
                const topLeftOffest = topRowOffset + sourceColFloor * imagesStrides[2];
                const botLeftOffset = botRowOffset + sourceColFloor * imagesStrides[2];
                const topRightOffset = topRowOffset + sourceColCeil * imagesStrides[2];
                const botRightOffest = botRowOffset + sourceColCeil * imagesStrides[2];
                for (let d = 0; d < numChannels; d++) {
                    // Begin shader.
                    // Compute the fractional index of the source.
                    const topLeft = xValues[topLeftOffest + d];
                    const bottomLeft = xValues[botLeftOffset + d];
                    const topRight = xValues[topRightOffset + d];
                    const bottomRight = xValues[botRightOffest + d];
                    const top = topLeft + (topRight - topLeft) * colFrac;
                    const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                    const newValue = top + (bottom - top) * rowFrac;
                    result[outputIdx++] = newValue;
                }
            }
        }
    }
    return backend.makeTensorInfo([batch, newHeight, newWidth, numChannels], 'float32', result);
}
export const resizeBilinearConfig = {
    kernelName: ResizeBilinear,
    backendName: 'cpu',
    kernelFunc: resizeBilinear
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUmVzaXplQmlsaW5lYXIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtY3B1L3NyYy9rZXJuZWxzL1Jlc2l6ZUJpbGluZWFyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBMkIsY0FBYyxFQUFxRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUd4SixPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFN0MsTUFBTSxVQUFVLGNBQWMsQ0FBQyxJQUk5QjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsTUFBTSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ3hCLE1BQU0sRUFBQyxZQUFZLEVBQUUsZ0JBQWdCLEVBQUUsSUFBSSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRXJELGdCQUFnQixDQUFDLE1BQU0sRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO0lBRTNDLE1BQU0sYUFBYSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQ3hELE1BQU0sQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLEdBQUcsSUFBSSxDQUFDO0lBRW5DLE1BQU0sQ0FBQyxLQUFLLEVBQUUsU0FBUyxFQUFFLFFBQVEsRUFBRSxXQUFXLENBQUMsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDO0lBQy9ELE1BQU0sT0FBTyxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxNQUFvQixDQUFDO0lBQ3JFLE1BQU0sTUFBTSxHQUFHLElBQUksWUFBWSxDQUMzQixJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsS0FBSyxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsV0FB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QUFDO2lCQUNoQzthQUNGO1NBQ0Y7S0FDRjtJQUVELE9BQU8sT0FBTyxDQUFDLGNBQWMsQ0FDekIsQ0FBQyxLQUFLLEVBQUUsU0FBUyxFQUFFLFFBQVEsRUFBRSxXQUFXLENBQUMsRUFBRSxTQUFTLEVBQUUsTUFBTSxDQUFDLENBQUM7QUFDcEUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLG9CQUFvQixHQUFpQjtJQUNoRCxVQUFVLEVBQUUsY0FBYztJQUMxQixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsY0FBdUM7Q0FDcEQsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFJlc2l6ZUJpbGluZWFyLCBSZXNpemVCaWxpbmVhckF0dHJzLCBSZXNpemVCaWxpbmVhcklucHV0cywgVGVuc29ySW5mbywgVHlwZWRBcnJheSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZENQVX0gZnJvbSAnLi4vYmFja2VuZF9jcHUnO1xuaW1wb3J0IHthc3NlcnROb3RDb21wbGV4fSBmcm9tICcuLi9jcHVfdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiByZXNpemVCaWxpbmVhcihhcmdzOiB7XG4gIGlucHV0czogUmVzaXplQmlsaW5lYXJJbnB1dHMsXG4gIGJhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLFxuICBhdHRyczogUmVzaXplQmlsaW5lYXJBdHRyc1xufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7aW1hZ2VzfSA9IGlucHV0cztcbiAgY29uc3Qge2FsaWduQ29ybmVycywgaGFsZlBpeGVsQ2VudGVycywgc2l6ZX0gPSBhdHRycztcblxuICBhc3NlcnROb3RDb21wbGV4KGltYWdlcywgJ3Jlc2l6ZUJpbGluZWFyJyk7XG5cbiAgY29uc3QgaW1hZ2VzU3RyaWRlcyA9IHV0aWwuY29tcHV0ZVN0cmlkZXMoaW1hZ2VzLnNoYXBlKTtcbiAgY29uc3QgW25ld0hlaWdodCwgbmV3V2lkdGhdID0gc2l6ZTtcblxuICBjb25zdCBbYmF0Y2gsIG9sZEhlaWdodCwgb2xkV2lkdGgsIG51bUNoYW5uZWxzXSA9IGltYWdlcy5zaGFwZTtcbiAgY29uc3QgeFZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQoaW1hZ2VzLmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG4gIGNvbnN0IHJlc3VsdCA9IG5ldyBGbG9hdDMyQXJyYXkoXG4gICAgICB1dGlsLnNpemVGcm9tU2hhcGUoW2JhdGNoLCBuZXdIZWlnaHQsIG5ld1dpZHRoLCBudW1DaGFubmVsc10pKTtcblxuICBjb25zdCBlZmZlY3RpdmVJbnB1dFNpemU6IFtudW1iZXIsIG51bWJlcl0gPSBbXG4gICAgKGFsaWduQ29ybmVycyAmJiBuZXdIZWlnaHQgPiAxKSA/IG9sZEhlaWdodCAtIDEgOiBvbGRIZWlnaHQsXG4gICAgKGFsaWduQ29ybmVycyAmJiBuZXdXaWR0aCA+IDEpID8gb2xkV2lkdGggLSAxIDogb2xkV2lkdGhcbiAgXTtcblxuICBjb25zdCBlZmZlY3RpdmVPdXRwdXRTaXplOiBbbnVtYmVyLCBudW1iZXJdID0gW1xuICAgIChhbGlnbkNvcm5lcnMgJiYgbmV3SGVpZ2h0ID4gMSkgPyBuZXdIZWlnaHQgLSAxIDogbmV3SGVpZ2h0LFxuICAgIChhbGlnbkNvcm5lcnMgJiYgbmV3V2lkdGggPiAxKSA/IG5ld1dpZHRoIC0gMSA6IG5ld1dpZHRoXG4gIF07XG4gIGxldCBvdXRwdXRJZHggPSAwO1xuICBjb25zdCBlZmZlY3RpdmVSb3dTaXplUmF0aW8gPSBlZmZlY3RpdmVJbnB1dFNpemVbMF0gLyBlZmZlY3RpdmVPdXRwdXRTaXplWzBdO1xuICBjb25zdCBlZmZlY3RpdmVDb2xTaXplUmF0aW8gPSBlZmZlY3RpdmVJbnB1dFNpemVbMV0gLyBlZmZlY3RpdmVPdXRwdXRTaXplWzFdO1xuICBmb3IgKGxldCBiID0gMDsgYiA8IGJhdGNoOyBiKyspIHtcbiAgICBmb3IgKGxldCByID0gMDsgciA8IG5ld0hlaWdodDsgcisrKSB7XG4gICAgICBsZXQgc291cmNlRnJhY1JvdzogbnVtYmVyO1xuICAgICAgaWYgKGhhbGZQaXhlbENlbnRlcnMpIHtcbiAgICAgICAgc291cmNlRnJhY1JvdyA9IGVmZmVjdGl2ZVJvd1NpemVSYXRpbyAqIChyICsgMC41KSAtIDAuNTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHNvdXJjZUZyYWNSb3cgPSBlZmZlY3RpdmVSb3dTaXplUmF0aW8gKiByO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBzb3VyY2VSb3dGbG9vciA9IE1hdGgubWF4KDAsIE1hdGguZmxvb3Ioc291cmNlRnJhY1JvdykpO1xuICAgICAgY29uc3Qgcm93RnJhYyA9IHNvdXJjZUZyYWNSb3cgLSBzb3VyY2VSb3dGbG9vcjtcbiAgICAgIGNvbnN0IHNvdXJjZVJvd0NlaWwgPSBNYXRoLm1pbihvbGRIZWlnaHQgLSAxLCBNYXRoLmNlaWwoc291cmNlRnJhY1JvdykpO1xuICAgICAgY29uc3QgdG9wUm93T2Zmc2V0ID1cbiAgICAgICAgICBiICogaW1hZ2VzU3RyaWRlc1swXSArIHNvdXJjZVJvd0Zsb29yICogaW1hZ2VzU3RyaWRlc1sxXTtcbiAgICAgIGNvbnN0IGJvdFJvd09mZnNldCA9XG4gICAgICAgICAgYiAqIGltYWdlc1N0cmlkZXNbMF0gKyBzb3VyY2VSb3dDZWlsICogaW1hZ2VzU3RyaWRlc1sxXTtcbiAgICAgIGZvciAobGV0IGMgPSAwOyBjIDwgbmV3V2lkdGg7IGMrKykge1xuICAgICAgICBsZXQgc291cmNlRnJhY0NvbDogbnVtYmVyO1xuICAgICAgICBpZiAoaGFsZlBpeGVsQ2VudGVycykge1xuICAgICAgICAgIHNvdXJjZUZyYWNDb2wgPSBlZmZlY3RpdmVDb2xTaXplUmF0aW8gKiAoYyArIDAuNSkgLSAwLjU7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgc291cmNlRnJhY0NvbCA9IGVmZmVjdGl2ZUNvbFNpemVSYXRpbyAqIGM7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3Qgc291cmNlQ29sRmxvb3IgPSBNYXRoLm1heCgwLCBNYXRoLmZsb29yKHNvdXJjZUZyYWNDb2wpKTtcbiAgICAgICAgY29uc3QgY29sRnJhYyA9IHNvdXJjZUZyYWNDb2wgLSBzb3VyY2VDb2xGbG9vcjtcbiAgICAgICAgY29uc3Qgc291cmNlQ29sQ2VpbCA9IE1hdGgubWluKG9sZFdpZHRoIC0gMSwgTWF0aC5jZWlsKHNvdXJjZUZyYWNDb2wpKTtcbiAgICAgICAgY29uc3QgdG9wTGVmdE9mZmVzdCA9IHRvcFJvd09mZnNldCArIHNvdXJjZUNvbEZsb29yICogaW1hZ2VzU3RyaWRlc1syXTtcbiAgICAgICAgY29uc3QgYm90TGVmdE9mZnNldCA9IGJvdFJvd09mZnNldCArIHNvdXJjZUNvbEZsb29yICogaW1hZ2VzU3RyaWRlc1syXTtcbiAgICAgICAgY29uc3QgdG9wUmlnaHRPZmZzZXQgPSB0b3BSb3dPZmZzZXQgKyBzb3VyY2VDb2xDZWlsICogaW1hZ2VzU3RyaWRlc1syXTtcbiAgICAgICAgY29uc3QgYm90UmlnaHRPZmZlc3QgPSBib3RSb3dPZmZzZXQgKyBzb3VyY2VDb2xDZWlsICogaW1hZ2VzU3RyaWRlc1syXTtcbiAgICAgICAgZm9yIChsZXQgZCA9IDA7IGQgPCBudW1DaGFubmVsczsgZCsrKSB7XG4gICAgICAgICAgLy8gQmVnaW4gc2hhZGVyLlxuXG4gICAgICAgICAgLy8gQ29tcHV0ZSB0aGUgZnJhY3Rpb25hbCBpbmRleCBvZiB0aGUgc291cmNlLlxuICAgICAgICAgIGNvbnN0IHRvcExlZnQgPSB4VmFsdWVzW3RvcExlZnRPZmZlc3QgKyBkXTtcbiAgICAgICAgICBjb25zdCBib3R0b21MZWZ0ID0geFZhbHVlc1tib3RMZWZ0T2Zmc2V0ICsgZF07XG4gICAgICAgICAgY29uc3QgdG9wUmlnaHQgPSB4VmFsdWVzW3RvcFJpZ2h0T2Zmc2V0ICsgZF07XG4gICAgICAgICAgY29uc3QgYm90dG9tUmlnaHQgPSB4VmFsdWVzW2JvdFJpZ2h0T2ZmZXN0ICsgZF07XG5cbiAgICAgICAgICBjb25zdCB0b3AgPSB0b3BMZWZ0ICsgKHRvcFJpZ2h0IC0gdG9wTGVmdCkgKiBjb2xGcmFjO1xuICAgICAgICAgIGNvbnN0IGJvdHRvbSA9IGJvdHRvbUxlZnQgKyAoYm90dG9tUmlnaHQgLSBib3R0b21MZWZ0KSAqIGNvbEZyYWM7XG4gICAgICAgICAgY29uc3QgbmV3VmFsdWUgPSB0b3AgKyAoYm90dG9tIC0gdG9wKSAqIHJvd0ZyYWM7XG5cbiAgICAgICAgICByZXN1bHRbb3V0cHV0SWR4KytdID0gbmV3VmFsdWU7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICByZXR1cm4gYmFja2VuZC5tYWtlVGVuc29ySW5mbyhcbiAgICAgIFtiYXRjaCwgbmV3SGVpZ2h0LCBuZXdXaWR0aCwgbnVtQ2hhbm5lbHNdLCAnZmxvYXQzMicsIHJlc3VsdCk7XG59XG5cbmV4cG9ydCBjb25zdCByZXNpemVCaWxpbmVhckNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBSZXNpemVCaWxpbmVhcixcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiByZXNpemVCaWxpbmVhciBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=