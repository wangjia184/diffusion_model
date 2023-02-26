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
import { SpaceToBatchND } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * This operation divides "spatial" dimensions `[1, ..., M]` of the input into
 * a grid of blocks of shape `blockShape`, and interleaves these blocks with
 * the "batch" dimension (0) such that in the output, the spatial
 * dimensions `[1, ..., M]` correspond to the position within the grid,
 * and the batch dimension combines both the position within a spatial block
 * and the original batch position. Prior to division into blocks,
 * the spatial dimensions of the input are optionally zero padded
 * according to `paddings`. See below for a precise description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
 * const blockShape = [2, 2];
 * const paddings = [[0, 0], [0, 0]];
 *
 * x.spaceToBatchND(blockShape, paddings).print();
 * ```
 *
 * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must have shape `[M]`, all values must
 * be >= 1.
 * @param paddings A 2-D array. Must have shape `[M, 2]`, all values must be >=
 *     0. `paddings[i] = [padStart, padEnd]` specifies the amount to zero-pad
 * from input dimension `i + 1`, which corresponds to spatial dimension `i`. It
 * is required that
 * `(inputShape[i + 1] + padStart + padEnd) % blockShape[i] === 0`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the input
 * according to `paddings` to produce `padded` of shape paddedShape.
 *
 * 2. Reshape `padded` to `reshapedPadded` of shape:
 * `[batch] + [paddedShape[1] / blockShape[0], blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1], blockShape[M-1]] + remainingShape`
 *
 * 3. Permute dimensions of `reshapedPadded` to produce `permutedReshapedPadded`
 * of shape: `blockShape + [batch] + [paddedShape[1] / blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1]] + remainingShape`
 *
 * 4. Reshape `permutedReshapedPadded` to flatten `blockShape` into the
 * batch dimension, producing an output tensor of shape:
 * `[batch * prod(blockShape)] + [paddedShape[1] / blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1]] + remainingShape`
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function spaceToBatchND_(x, blockShape, paddings) {
    const $x = convertToTensor(x, 'x', 'spaceToBatchND');
    util.assert($x.rank >= 1 + blockShape.length, () => `input rank ${$x.rank} should be > than [blockShape] ${blockShape.length}`);
    util.assert(paddings.length === blockShape.length, () => `paddings.shape[0] ${paddings.length} must be equal to [blockShape] ${blockShape.length}`);
    util.assert($x.shape.reduce((a, b, i) => {
        if (i > 0 && i <= blockShape.length) {
            return a &&
                ((b + paddings[i - 1][0] + paddings[i - 1][1]) %
                    blockShape[i - 1] ===
                    0);
        }
        return a;
    }, true), () => `input spatial dimensions ${$x.shape.slice(1)} with paddings ${paddings.toString()} must be divisible by blockShapes ${blockShape.toString()}`);
    const inputs = { x: $x };
    const attrs = { blockShape, paddings };
    return ENGINE.runKernel(SpaceToBatchND, inputs, attrs);
}
export const spaceToBatchND = /* @__PURE__ */ op({ spaceToBatchND_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhY2VfdG9fYmF0Y2hfbmQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zcGFjZV90b19iYXRjaF9uZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxjQUFjLEVBQTRDLE1BQU0saUJBQWlCLENBQUM7QUFJMUYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBK0NHO0FBQ0gsU0FBUyxlQUFlLENBQ3BCLENBQWUsRUFBRSxVQUFvQixFQUFFLFFBQW9CO0lBQzdELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLGdCQUFnQixDQUFDLENBQUM7SUFFckQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxJQUFJLENBQUMsR0FBRyxVQUFVLENBQUMsTUFBTSxFQUNoQyxHQUFHLEVBQUUsQ0FBQyxjQUFjLEVBQUUsQ0FBQyxJQUFJLGtDQUN2QixVQUFVLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQztJQUU3QixJQUFJLENBQUMsTUFBTSxDQUNQLFFBQVEsQ0FBQyxNQUFNLEtBQUssVUFBVSxDQUFDLE1BQU0sRUFDckMsR0FBRyxFQUFFLENBQUMscUJBQ0YsUUFBUSxDQUFDLE1BQU0sa0NBQWtDLFVBQVUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDO0lBRTlFLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQ1gsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQ1YsSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsSUFBSSxVQUFVLENBQUMsTUFBTSxFQUFFO1lBQ25DLE9BQU8sQ0FBQztnQkFDSixDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDekMsVUFBVSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7b0JBQ3JCLENBQUMsQ0FBQyxDQUFDO1NBQ1Q7UUFDRCxPQUFPLENBQUMsQ0FBQztJQUNYLENBQUMsRUFDRCxJQUFJLENBQUMsRUFDVCxHQUFHLEVBQUUsQ0FBQyw0QkFBNEIsRUFBRSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLGtCQUMvQyxRQUFRLENBQUMsUUFBUSxFQUFFLHFDQUNuQixVQUFVLENBQUMsUUFBUSxFQUFFLEVBQUUsQ0FBQyxDQUFDO0lBRWpDLE1BQU0sTUFBTSxHQUF5QixFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUM3QyxNQUFNLEtBQUssR0FBd0IsRUFBQyxVQUFVLEVBQUUsUUFBUSxFQUFDLENBQUM7SUFFMUQsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixjQUFjLEVBQUUsTUFBbUMsRUFDbkQsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxjQUFjLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGVBQWUsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtTcGFjZVRvQmF0Y2hORCwgU3BhY2VUb0JhdGNoTkRBdHRycywgU3BhY2VUb0JhdGNoTkRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogVGhpcyBvcGVyYXRpb24gZGl2aWRlcyBcInNwYXRpYWxcIiBkaW1lbnNpb25zIGBbMSwgLi4uLCBNXWAgb2YgdGhlIGlucHV0IGludG9cbiAqIGEgZ3JpZCBvZiBibG9ja3Mgb2Ygc2hhcGUgYGJsb2NrU2hhcGVgLCBhbmQgaW50ZXJsZWF2ZXMgdGhlc2UgYmxvY2tzIHdpdGhcbiAqIHRoZSBcImJhdGNoXCIgZGltZW5zaW9uICgwKSBzdWNoIHRoYXQgaW4gdGhlIG91dHB1dCwgdGhlIHNwYXRpYWxcbiAqIGRpbWVuc2lvbnMgYFsxLCAuLi4sIE1dYCBjb3JyZXNwb25kIHRvIHRoZSBwb3NpdGlvbiB3aXRoaW4gdGhlIGdyaWQsXG4gKiBhbmQgdGhlIGJhdGNoIGRpbWVuc2lvbiBjb21iaW5lcyBib3RoIHRoZSBwb3NpdGlvbiB3aXRoaW4gYSBzcGF0aWFsIGJsb2NrXG4gKiBhbmQgdGhlIG9yaWdpbmFsIGJhdGNoIHBvc2l0aW9uLiBQcmlvciB0byBkaXZpc2lvbiBpbnRvIGJsb2NrcyxcbiAqIHRoZSBzcGF0aWFsIGRpbWVuc2lvbnMgb2YgdGhlIGlucHV0IGFyZSBvcHRpb25hbGx5IHplcm8gcGFkZGVkXG4gKiBhY2NvcmRpbmcgdG8gYHBhZGRpbmdzYC4gU2VlIGJlbG93IGZvciBhIHByZWNpc2UgZGVzY3JpcHRpb24uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChbMSwgMiwgMywgNF0sIFsxLCAyLCAyLCAxXSk7XG4gKiBjb25zdCBibG9ja1NoYXBlID0gWzIsIDJdO1xuICogY29uc3QgcGFkZGluZ3MgPSBbWzAsIDBdLCBbMCwgMF1dO1xuICpcbiAqIHguc3BhY2VUb0JhdGNoTkQoYmxvY2tTaGFwZSwgcGFkZGluZ3MpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBBIGB0Zi5UZW5zb3JgLiBOLUQgd2l0aCBgeC5zaGFwZWAgPSBgW2JhdGNoXSArIHNwYXRpYWxTaGFwZSArXG4gKiByZW1haW5pbmdTaGFwZWAsIHdoZXJlIHNwYXRpYWxTaGFwZSBoYXMgYE1gIGRpbWVuc2lvbnMuXG4gKiBAcGFyYW0gYmxvY2tTaGFwZSBBIDEtRCBhcnJheS4gTXVzdCBoYXZlIHNoYXBlIGBbTV1gLCBhbGwgdmFsdWVzIG11c3RcbiAqIGJlID49IDEuXG4gKiBAcGFyYW0gcGFkZGluZ3MgQSAyLUQgYXJyYXkuIE11c3QgaGF2ZSBzaGFwZSBgW00sIDJdYCwgYWxsIHZhbHVlcyBtdXN0IGJlID49XG4gKiAgICAgMC4gYHBhZGRpbmdzW2ldID0gW3BhZFN0YXJ0LCBwYWRFbmRdYCBzcGVjaWZpZXMgdGhlIGFtb3VudCB0byB6ZXJvLXBhZFxuICogZnJvbSBpbnB1dCBkaW1lbnNpb24gYGkgKyAxYCwgd2hpY2ggY29ycmVzcG9uZHMgdG8gc3BhdGlhbCBkaW1lbnNpb24gYGlgLiBJdFxuICogaXMgcmVxdWlyZWQgdGhhdFxuICogYChpbnB1dFNoYXBlW2kgKyAxXSArIHBhZFN0YXJ0ICsgcGFkRW5kKSAlIGJsb2NrU2hhcGVbaV0gPT09IDBgXG4gKlxuICogVGhpcyBvcGVyYXRpb24gaXMgZXF1aXZhbGVudCB0byB0aGUgZm9sbG93aW5nIHN0ZXBzOlxuICpcbiAqIDEuIFplcm8tcGFkIHRoZSBzdGFydCBhbmQgZW5kIG9mIGRpbWVuc2lvbnMgYFsxLCAuLi4sIE1dYCBvZiB0aGUgaW5wdXRcbiAqIGFjY29yZGluZyB0byBgcGFkZGluZ3NgIHRvIHByb2R1Y2UgYHBhZGRlZGAgb2Ygc2hhcGUgcGFkZGVkU2hhcGUuXG4gKlxuICogMi4gUmVzaGFwZSBgcGFkZGVkYCB0byBgcmVzaGFwZWRQYWRkZWRgIG9mIHNoYXBlOlxuICogYFtiYXRjaF0gKyBbcGFkZGVkU2hhcGVbMV0gLyBibG9ja1NoYXBlWzBdLCBibG9ja1NoYXBlWzBdLCAuLi4sXG4gKiBwYWRkZWRTaGFwZVtNXSAvIGJsb2NrU2hhcGVbTS0xXSwgYmxvY2tTaGFwZVtNLTFdXSArIHJlbWFpbmluZ1NoYXBlYFxuICpcbiAqIDMuIFBlcm11dGUgZGltZW5zaW9ucyBvZiBgcmVzaGFwZWRQYWRkZWRgIHRvIHByb2R1Y2UgYHBlcm11dGVkUmVzaGFwZWRQYWRkZWRgXG4gKiBvZiBzaGFwZTogYGJsb2NrU2hhcGUgKyBbYmF0Y2hdICsgW3BhZGRlZFNoYXBlWzFdIC8gYmxvY2tTaGFwZVswXSwgLi4uLFxuICogcGFkZGVkU2hhcGVbTV0gLyBibG9ja1NoYXBlW00tMV1dICsgcmVtYWluaW5nU2hhcGVgXG4gKlxuICogNC4gUmVzaGFwZSBgcGVybXV0ZWRSZXNoYXBlZFBhZGRlZGAgdG8gZmxhdHRlbiBgYmxvY2tTaGFwZWAgaW50byB0aGVcbiAqIGJhdGNoIGRpbWVuc2lvbiwgcHJvZHVjaW5nIGFuIG91dHB1dCB0ZW5zb3Igb2Ygc2hhcGU6XG4gKiBgW2JhdGNoICogcHJvZChibG9ja1NoYXBlKV0gKyBbcGFkZGVkU2hhcGVbMV0gLyBibG9ja1NoYXBlWzBdLCAuLi4sXG4gKiBwYWRkZWRTaGFwZVtNXSAvIGJsb2NrU2hhcGVbTS0xXV0gKyByZW1haW5pbmdTaGFwZWBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdUcmFuc2Zvcm1hdGlvbnMnfVxuICovXG5mdW5jdGlvbiBzcGFjZVRvQmF0Y2hORF88VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBibG9ja1NoYXBlOiBudW1iZXJbXSwgcGFkZGluZ3M6IG51bWJlcltdW10pOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnc3BhY2VUb0JhdGNoTkQnKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICR4LnJhbmsgPj0gMSArIGJsb2NrU2hhcGUubGVuZ3RoLFxuICAgICAgKCkgPT4gYGlucHV0IHJhbmsgJHskeC5yYW5rfSBzaG91bGQgYmUgPiB0aGFuIFtibG9ja1NoYXBlXSAke1xuICAgICAgICAgIGJsb2NrU2hhcGUubGVuZ3RofWApO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgcGFkZGluZ3MubGVuZ3RoID09PSBibG9ja1NoYXBlLmxlbmd0aCxcbiAgICAgICgpID0+IGBwYWRkaW5ncy5zaGFwZVswXSAke1xuICAgICAgICAgIHBhZGRpbmdzLmxlbmd0aH0gbXVzdCBiZSBlcXVhbCB0byBbYmxvY2tTaGFwZV0gJHtibG9ja1NoYXBlLmxlbmd0aH1gKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICR4LnNoYXBlLnJlZHVjZShcbiAgICAgICAgICAoYSwgYiwgaSkgPT4ge1xuICAgICAgICAgICAgaWYgKGkgPiAwICYmIGkgPD0gYmxvY2tTaGFwZS5sZW5ndGgpIHtcbiAgICAgICAgICAgICAgcmV0dXJuIGEgJiZcbiAgICAgICAgICAgICAgICAgICgoYiArIHBhZGRpbmdzW2kgLSAxXVswXSArIHBhZGRpbmdzW2kgLSAxXVsxXSkgJVxuICAgICAgICAgICAgICAgICAgICAgICBibG9ja1NoYXBlW2kgLSAxXSA9PT1cbiAgICAgICAgICAgICAgICAgICAwKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBhO1xuICAgICAgICAgIH0sXG4gICAgICAgICAgdHJ1ZSksXG4gICAgICAoKSA9PiBgaW5wdXQgc3BhdGlhbCBkaW1lbnNpb25zICR7JHguc2hhcGUuc2xpY2UoMSl9IHdpdGggcGFkZGluZ3MgJHtcbiAgICAgICAgICBwYWRkaW5ncy50b1N0cmluZygpfSBtdXN0IGJlIGRpdmlzaWJsZSBieSBibG9ja1NoYXBlcyAke1xuICAgICAgICAgIGJsb2NrU2hhcGUudG9TdHJpbmcoKX1gKTtcblxuICBjb25zdCBpbnB1dHM6IFNwYWNlVG9CYXRjaE5ESW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IFNwYWNlVG9CYXRjaE5EQXR0cnMgPSB7YmxvY2tTaGFwZSwgcGFkZGluZ3N9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgU3BhY2VUb0JhdGNoTkQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3Qgc3BhY2VUb0JhdGNoTkQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3NwYWNlVG9CYXRjaE5EX30pO1xuIl19