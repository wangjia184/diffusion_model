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
import { ENGINE } from '../engine';
import { BatchToSpaceND } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of
 * shape `blockShape + [batch]`, interleaves these blocks back into the grid
 * defined by the spatial dimensions `[1, ..., M]`, to obtain a result with
 * the same rank as the input. The spatial dimensions of this intermediate
 * result are then optionally cropped according to `crops` to produce the
 * output. This is the reverse of `tf.spaceToBatchND`. See below for a precise
 * description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
 * const blockShape = [2, 2];
 * const crops = [[0, 0], [0, 0]];
 *
 * x.batchToSpaceND(blockShape, crops).print();
 * ```
 *
 * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must have shape `[M]`, all values must
 * be >= 1.
 * @param crops A 2-D array.  Must have shape `[M, 2]`, all values must be >= 0.
 * `crops[i] = [cropStart, cropEnd]` specifies the amount to crop from input
 * dimension `i + 1`, which corresponds to spatial dimension `i`. It is required
 * that `cropStart[i] + cropEnd[i] <= blockShape[i] * inputShape[i + 1]`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Reshape `x` to `reshaped` of shape: `[blockShape[0], ...,
 * blockShape[M-1], batch / prod(blockShape), x.shape[1], ...,
 * x.shape[N-1]]`
 *
 * 2. Permute dimensions of `reshaped` to produce `permuted` of shape `[batch /
 * prod(blockShape),x.shape[1], blockShape[0], ..., x.shape[M],
 * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * 3. Reshape `permuted` to produce `reshapedPermuted` of shape `[batch /
 * prod(blockShape),x.shape[1] * blockShape[0], ..., x.shape[M] *
 * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * 4. Crop the start and end of dimensions `[1, ..., M]` of `reshapedPermuted`
 * according to `crops` to produce the output of shape: `[batch /
 * prod(blockShape),x.shape[1] * blockShape[0] - crops[0,0] - crops[0,1],
 * ..., x.shape[M] * blockShape[M-1] - crops[M-1,0] -
 * crops[M-1,1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function batchToSpaceND_(x, blockShape, crops) {
    const $x = convertToTensor(x, 'x', 'batchToSpaceND');
    const prod = blockShape.reduce((a, b) => a * b);
    util.assert($x.rank >= 1 + blockShape.length, () => `input rank is ${$x.rank} but should be > than blockShape.length ${blockShape.length}`);
    util.assert(crops.length === blockShape.length, () => `crops.length is ${crops.length} but should be equal to blockShape.length  ${blockShape.length}`);
    util.assert($x.shape[0] % prod === 0, () => `input tensor batch is ${$x.shape[0]} but is not divisible by the product of ` +
        `the elements of blockShape ${blockShape.join(' * ')} === ${prod}`);
    const inputs = { x: $x };
    const attrs = { blockShape, crops };
    return ENGINE.runKernel(BatchToSpaceND, inputs, attrs);
}
export const batchToSpaceND = /* @__PURE__ */ op({ batchToSpaceND_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmF0Y2hfdG9fc3BhY2VfbmQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9iYXRjaF90b19zcGFjZV9uZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxjQUFjLEVBQTRDLE1BQU0saUJBQWlCLENBQUM7QUFJMUYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBK0NHO0FBQ0gsU0FBUyxlQUFlLENBQ3BCLENBQWUsRUFBRSxVQUFvQixFQUFFLEtBQWlCO0lBQzFELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLGdCQUFnQixDQUFDLENBQUM7SUFDckQsTUFBTSxJQUFJLEdBQUcsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUVoRCxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLElBQUksQ0FBQyxHQUFHLFVBQVUsQ0FBQyxNQUFNLEVBQ2hDLEdBQUcsRUFBRSxDQUFDLGlCQUFpQixFQUFFLENBQUMsSUFBSSwyQ0FDMUIsVUFBVSxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUM7SUFFN0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsTUFBTSxLQUFLLFVBQVUsQ0FBQyxNQUFNLEVBQ2xDLEdBQUcsRUFBRSxDQUFDLG1CQUNGLEtBQUssQ0FBQyxNQUFNLDhDQUNaLFVBQVUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDO0lBRTdCLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEtBQUssQ0FBQyxFQUN4QixHQUFHLEVBQUUsQ0FBQyx5QkFDSSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQywwQ0FBMEM7UUFDM0QsOEJBQThCLFVBQVUsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsSUFBSSxFQUFFLENBQUMsQ0FBQztJQUU1RSxNQUFNLE1BQU0sR0FBeUIsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDN0MsTUFBTSxLQUFLLEdBQXdCLEVBQUMsVUFBVSxFQUFFLEtBQUssRUFBQyxDQUFDO0lBRXZELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsY0FBYyxFQUFFLE1BQW1DLEVBQ25ELEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sY0FBYyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxlQUFlLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7QmF0Y2hUb1NwYWNlTkQsIEJhdGNoVG9TcGFjZU5EQXR0cnMsIEJhdGNoVG9TcGFjZU5ESW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFRoaXMgb3BlcmF0aW9uIHJlc2hhcGVzIHRoZSBcImJhdGNoXCIgZGltZW5zaW9uIDAgaW50byBgTSArIDFgIGRpbWVuc2lvbnMgb2ZcbiAqIHNoYXBlIGBibG9ja1NoYXBlICsgW2JhdGNoXWAsIGludGVybGVhdmVzIHRoZXNlIGJsb2NrcyBiYWNrIGludG8gdGhlIGdyaWRcbiAqIGRlZmluZWQgYnkgdGhlIHNwYXRpYWwgZGltZW5zaW9ucyBgWzEsIC4uLiwgTV1gLCB0byBvYnRhaW4gYSByZXN1bHQgd2l0aFxuICogdGhlIHNhbWUgcmFuayBhcyB0aGUgaW5wdXQuIFRoZSBzcGF0aWFsIGRpbWVuc2lvbnMgb2YgdGhpcyBpbnRlcm1lZGlhdGVcbiAqIHJlc3VsdCBhcmUgdGhlbiBvcHRpb25hbGx5IGNyb3BwZWQgYWNjb3JkaW5nIHRvIGBjcm9wc2AgdG8gcHJvZHVjZSB0aGVcbiAqIG91dHB1dC4gVGhpcyBpcyB0aGUgcmV2ZXJzZSBvZiBgdGYuc3BhY2VUb0JhdGNoTkRgLiBTZWUgYmVsb3cgZm9yIGEgcHJlY2lzZVxuICogZGVzY3JpcHRpb24uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChbMSwgMiwgMywgNF0sIFs0LCAxLCAxLCAxXSk7XG4gKiBjb25zdCBibG9ja1NoYXBlID0gWzIsIDJdO1xuICogY29uc3QgY3JvcHMgPSBbWzAsIDBdLCBbMCwgMF1dO1xuICpcbiAqIHguYmF0Y2hUb1NwYWNlTkQoYmxvY2tTaGFwZSwgY3JvcHMpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBBIGB0Zi5UZW5zb3JgLiBOLUQgd2l0aCBgeC5zaGFwZWAgPSBgW2JhdGNoXSArIHNwYXRpYWxTaGFwZSArXG4gKiByZW1haW5pbmdTaGFwZWAsIHdoZXJlIHNwYXRpYWxTaGFwZSBoYXMgYE1gIGRpbWVuc2lvbnMuXG4gKiBAcGFyYW0gYmxvY2tTaGFwZSBBIDEtRCBhcnJheS4gTXVzdCBoYXZlIHNoYXBlIGBbTV1gLCBhbGwgdmFsdWVzIG11c3RcbiAqIGJlID49IDEuXG4gKiBAcGFyYW0gY3JvcHMgQSAyLUQgYXJyYXkuICBNdXN0IGhhdmUgc2hhcGUgYFtNLCAyXWAsIGFsbCB2YWx1ZXMgbXVzdCBiZSA+PSAwLlxuICogYGNyb3BzW2ldID0gW2Nyb3BTdGFydCwgY3JvcEVuZF1gIHNwZWNpZmllcyB0aGUgYW1vdW50IHRvIGNyb3AgZnJvbSBpbnB1dFxuICogZGltZW5zaW9uIGBpICsgMWAsIHdoaWNoIGNvcnJlc3BvbmRzIHRvIHNwYXRpYWwgZGltZW5zaW9uIGBpYC4gSXQgaXMgcmVxdWlyZWRcbiAqIHRoYXQgYGNyb3BTdGFydFtpXSArIGNyb3BFbmRbaV0gPD0gYmxvY2tTaGFwZVtpXSAqIGlucHV0U2hhcGVbaSArIDFdYFxuICpcbiAqIFRoaXMgb3BlcmF0aW9uIGlzIGVxdWl2YWxlbnQgdG8gdGhlIGZvbGxvd2luZyBzdGVwczpcbiAqXG4gKiAxLiBSZXNoYXBlIGB4YCB0byBgcmVzaGFwZWRgIG9mIHNoYXBlOiBgW2Jsb2NrU2hhcGVbMF0sIC4uLixcbiAqIGJsb2NrU2hhcGVbTS0xXSwgYmF0Y2ggLyBwcm9kKGJsb2NrU2hhcGUpLCB4LnNoYXBlWzFdLCAuLi4sXG4gKiB4LnNoYXBlW04tMV1dYFxuICpcbiAqIDIuIFBlcm11dGUgZGltZW5zaW9ucyBvZiBgcmVzaGFwZWRgIHRvIHByb2R1Y2UgYHBlcm11dGVkYCBvZiBzaGFwZSBgW2JhdGNoIC9cbiAqIHByb2QoYmxvY2tTaGFwZSkseC5zaGFwZVsxXSwgYmxvY2tTaGFwZVswXSwgLi4uLCB4LnNoYXBlW01dLFxuICogYmxvY2tTaGFwZVtNLTFdLHguc2hhcGVbTSsxXSwgLi4uLCB4LnNoYXBlW04tMV1dYFxuICpcbiAqIDMuIFJlc2hhcGUgYHBlcm11dGVkYCB0byBwcm9kdWNlIGByZXNoYXBlZFBlcm11dGVkYCBvZiBzaGFwZSBgW2JhdGNoIC9cbiAqIHByb2QoYmxvY2tTaGFwZSkseC5zaGFwZVsxXSAqIGJsb2NrU2hhcGVbMF0sIC4uLiwgeC5zaGFwZVtNXSAqXG4gKiBibG9ja1NoYXBlW00tMV0seC5zaGFwZVtNKzFdLCAuLi4sIHguc2hhcGVbTi0xXV1gXG4gKlxuICogNC4gQ3JvcCB0aGUgc3RhcnQgYW5kIGVuZCBvZiBkaW1lbnNpb25zIGBbMSwgLi4uLCBNXWAgb2YgYHJlc2hhcGVkUGVybXV0ZWRgXG4gKiBhY2NvcmRpbmcgdG8gYGNyb3BzYCB0byBwcm9kdWNlIHRoZSBvdXRwdXQgb2Ygc2hhcGU6IGBbYmF0Y2ggL1xuICogcHJvZChibG9ja1NoYXBlKSx4LnNoYXBlWzFdICogYmxvY2tTaGFwZVswXSAtIGNyb3BzWzAsMF0gLSBjcm9wc1swLDFdLFxuICogLi4uLCB4LnNoYXBlW01dICogYmxvY2tTaGFwZVtNLTFdIC0gY3JvcHNbTS0xLDBdIC1cbiAqIGNyb3BzW00tMSwxXSx4LnNoYXBlW00rMV0sIC4uLiwgeC5zaGFwZVtOLTFdXWBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdUcmFuc2Zvcm1hdGlvbnMnfVxuICovXG5mdW5jdGlvbiBiYXRjaFRvU3BhY2VORF88VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBibG9ja1NoYXBlOiBudW1iZXJbXSwgY3JvcHM6IG51bWJlcltdW10pOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnYmF0Y2hUb1NwYWNlTkQnKTtcbiAgY29uc3QgcHJvZCA9IGJsb2NrU2hhcGUucmVkdWNlKChhLCBiKSA9PiBhICogYik7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkeC5yYW5rID49IDEgKyBibG9ja1NoYXBlLmxlbmd0aCxcbiAgICAgICgpID0+IGBpbnB1dCByYW5rIGlzICR7JHgucmFua30gYnV0IHNob3VsZCBiZSA+IHRoYW4gYmxvY2tTaGFwZS5sZW5ndGggJHtcbiAgICAgICAgICBibG9ja1NoYXBlLmxlbmd0aH1gKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgIGNyb3BzLmxlbmd0aCA9PT0gYmxvY2tTaGFwZS5sZW5ndGgsXG4gICAgICAoKSA9PiBgY3JvcHMubGVuZ3RoIGlzICR7XG4gICAgICAgICAgY3JvcHMubGVuZ3RofSBidXQgc2hvdWxkIGJlIGVxdWFsIHRvIGJsb2NrU2hhcGUubGVuZ3RoICAke1xuICAgICAgICAgIGJsb2NrU2hhcGUubGVuZ3RofWApO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHguc2hhcGVbMF0gJSBwcm9kID09PSAwLFxuICAgICAgKCkgPT4gYGlucHV0IHRlbnNvciBiYXRjaCBpcyAke1xuICAgICAgICAgICAgICAgICR4LnNoYXBlWzBdfSBidXQgaXMgbm90IGRpdmlzaWJsZSBieSB0aGUgcHJvZHVjdCBvZiBgICtcbiAgICAgICAgICBgdGhlIGVsZW1lbnRzIG9mIGJsb2NrU2hhcGUgJHtibG9ja1NoYXBlLmpvaW4oJyAqICcpfSA9PT0gJHtwcm9kfWApO1xuXG4gIGNvbnN0IGlucHV0czogQmF0Y2hUb1NwYWNlTkRJbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczogQmF0Y2hUb1NwYWNlTkRBdHRycyA9IHtibG9ja1NoYXBlLCBjcm9wc307XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBCYXRjaFRvU3BhY2VORCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBiYXRjaFRvU3BhY2VORCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7YmF0Y2hUb1NwYWNlTkRffSk7XG4iXX0=