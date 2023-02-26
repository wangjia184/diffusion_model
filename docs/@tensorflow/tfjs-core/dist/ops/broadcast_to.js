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
import { Tile } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { assertNonNegativeIntegerDimensions } from '../util_base';
import { clone } from './clone';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Broadcast an array to a compatible shape NumPy-style.
 *
 * The tensor's shape is compared to the broadcast shape from end to beginning.
 * Ones are prepended to the tensor's shape until it has the same length as
 * the broadcast shape. If input.shape[i]==shape[i], the (i+1)-th axis is
 * already broadcast-compatible. If input.shape[i]==1 and shape[i]==N, then
 * the input tensor is tiled N times along that axis (using tf.tile).
 *
 * @param input The tensor that is to be broadcasted.
 * @param shape The input is to be broadcast to this shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function broadcastTo_(x, shape) {
    let input = convertToTensor(x, 'broadcastTo', 'x');
    const xShape = input.shape;
    assertNonNegativeIntegerDimensions(shape);
    if (shape.length < input.rank) {
        throw new Error(`broadcastTo(): shape.length=${shape.length} < input.rank=${input.rank}.`);
    }
    if (shape.length > input.rank) {
        const newShape = input.shape.slice();
        while (newShape.length < shape.length) {
            newShape.unshift(1);
        }
        input = reshape(input, newShape);
    }
    const inputShape = input.shape;
    const reps = Array.from(shape);
    for (let i = shape.length - 1; i >= 0; i--) {
        if (inputShape[i] === shape[i]) {
            reps[i] = 1;
        }
        else if (input.shape[i] !== 1) {
            throw new Error(`broadcastTo(): [${xShape}] cannot be broadcast to [${shape}].`);
        }
    }
    const axes = reps.map((n, i) => n > 1 ? i : -1).filter(i => i >= 0);
    if (axes.length === 0) {
        return clone(input);
    }
    // TODO call broadcastTo kernel directly once backends implement broadcstTo
    const inputs = { x: input };
    const attrs = { reps };
    return ENGINE.runKernel(Tile, inputs, attrs);
}
export const broadcastTo = /* @__PURE__ */ op({ broadcastTo_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYnJvYWRjYXN0X3RvLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvYnJvYWRjYXN0X3RvLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLElBQUksRUFBd0IsTUFBTSxpQkFBaUIsQ0FBQztBQUk1RCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLGtDQUFrQyxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhFLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDOzs7Ozs7Ozs7Ozs7O0dBYUc7QUFDSCxTQUFTLFlBQVksQ0FDakIsQ0FBb0IsRUFBRSxLQUFrQjtJQUMxQyxJQUFJLEtBQUssR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLGFBQWEsRUFBRSxHQUFHLENBQUMsQ0FBQztJQUNuRCxNQUFNLE1BQU0sR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDO0lBRTNCLGtDQUFrQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRTFDLElBQUksS0FBSyxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUMsSUFBSSxFQUFFO1FBQzdCLE1BQU0sSUFBSSxLQUFLLENBQUMsK0JBQStCLEtBQUssQ0FBQyxNQUFNLGlCQUN2RCxLQUFLLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztLQUNwQjtJQUVELElBQUksS0FBSyxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUMsSUFBSSxFQUFFO1FBQzdCLE1BQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUMsS0FBSyxFQUFFLENBQUM7UUFDckMsT0FBTyxRQUFRLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQyxNQUFNLEVBQUU7WUFDckMsUUFBUSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUNyQjtRQUNELEtBQUssR0FBRyxPQUFPLENBQUMsS0FBSyxFQUFFLFFBQVEsQ0FBQyxDQUFDO0tBQ2xDO0lBRUQsTUFBTSxVQUFVLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQztJQUMvQixNQUFNLElBQUksR0FBYSxLQUFLLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQ3pDLEtBQUssSUFBSSxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUMxQyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsS0FBSyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUU7WUFDOUIsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQztTQUNiO2FBQU0sSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFBRTtZQUMvQixNQUFNLElBQUksS0FBSyxDQUNYLG1CQUFtQixNQUFNLDZCQUE2QixLQUFLLElBQUksQ0FBQyxDQUFDO1NBQ3RFO0tBQ0Y7SUFDRCxNQUFNLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUVwRSxJQUFJLElBQUksQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3JCLE9BQU8sS0FBSyxDQUFDLEtBQUssQ0FBYyxDQUFDO0tBQ2xDO0lBRUQsMkVBQTJFO0lBQzNFLE1BQU0sTUFBTSxHQUFlLEVBQUMsQ0FBQyxFQUFFLEtBQUssRUFBQyxDQUFDO0lBQ3RDLE1BQU0sS0FBSyxHQUFjLEVBQUMsSUFBSSxFQUFDLENBQUM7SUFDaEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixJQUFJLEVBQUUsTUFBbUMsRUFDekMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtUaWxlLCBUaWxlQXR0cnMsIFRpbGVJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtSYW5rLCBTaGFwZU1hcCwgVGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zfSBmcm9tICcuLi91dGlsX2Jhc2UnO1xuXG5pbXBvcnQge2Nsb25lfSBmcm9tICcuL2Nsb25lJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcblxuLyoqXG4gKiBCcm9hZGNhc3QgYW4gYXJyYXkgdG8gYSBjb21wYXRpYmxlIHNoYXBlIE51bVB5LXN0eWxlLlxuICpcbiAqIFRoZSB0ZW5zb3IncyBzaGFwZSBpcyBjb21wYXJlZCB0byB0aGUgYnJvYWRjYXN0IHNoYXBlIGZyb20gZW5kIHRvIGJlZ2lubmluZy5cbiAqIE9uZXMgYXJlIHByZXBlbmRlZCB0byB0aGUgdGVuc29yJ3Mgc2hhcGUgdW50aWwgaXQgaGFzIHRoZSBzYW1lIGxlbmd0aCBhc1xuICogdGhlIGJyb2FkY2FzdCBzaGFwZS4gSWYgaW5wdXQuc2hhcGVbaV09PXNoYXBlW2ldLCB0aGUgKGkrMSktdGggYXhpcyBpc1xuICogYWxyZWFkeSBicm9hZGNhc3QtY29tcGF0aWJsZS4gSWYgaW5wdXQuc2hhcGVbaV09PTEgYW5kIHNoYXBlW2ldPT1OLCB0aGVuXG4gKiB0aGUgaW5wdXQgdGVuc29yIGlzIHRpbGVkIE4gdGltZXMgYWxvbmcgdGhhdCBheGlzICh1c2luZyB0Zi50aWxlKS5cbiAqXG4gKiBAcGFyYW0gaW5wdXQgVGhlIHRlbnNvciB0aGF0IGlzIHRvIGJlIGJyb2FkY2FzdGVkLlxuICogQHBhcmFtIHNoYXBlIFRoZSBpbnB1dCBpcyB0byBiZSBicm9hZGNhc3QgdG8gdGhpcyBzaGFwZS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdUcmFuc2Zvcm1hdGlvbnMnfVxuICovXG5mdW5jdGlvbiBicm9hZGNhc3RUb188UiBleHRlbmRzIFJhbms+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBzaGFwZTogU2hhcGVNYXBbUl0pOiBUZW5zb3I8Uj4ge1xuICBsZXQgaW5wdXQgPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ2Jyb2FkY2FzdFRvJywgJ3gnKTtcbiAgY29uc3QgeFNoYXBlID0gaW5wdXQuc2hhcGU7XG5cbiAgYXNzZXJ0Tm9uTmVnYXRpdmVJbnRlZ2VyRGltZW5zaW9ucyhzaGFwZSk7XG5cbiAgaWYgKHNoYXBlLmxlbmd0aCA8IGlucHV0LnJhbmspIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYGJyb2FkY2FzdFRvKCk6IHNoYXBlLmxlbmd0aD0ke3NoYXBlLmxlbmd0aH0gPCBpbnB1dC5yYW5rPSR7XG4gICAgICAgIGlucHV0LnJhbmt9LmApO1xuICB9XG5cbiAgaWYgKHNoYXBlLmxlbmd0aCA+IGlucHV0LnJhbmspIHtcbiAgICBjb25zdCBuZXdTaGFwZSA9IGlucHV0LnNoYXBlLnNsaWNlKCk7XG4gICAgd2hpbGUgKG5ld1NoYXBlLmxlbmd0aCA8IHNoYXBlLmxlbmd0aCkge1xuICAgICAgbmV3U2hhcGUudW5zaGlmdCgxKTtcbiAgICB9XG4gICAgaW5wdXQgPSByZXNoYXBlKGlucHV0LCBuZXdTaGFwZSk7XG4gIH1cblxuICBjb25zdCBpbnB1dFNoYXBlID0gaW5wdXQuc2hhcGU7XG4gIGNvbnN0IHJlcHM6IG51bWJlcltdID0gQXJyYXkuZnJvbShzaGFwZSk7XG4gIGZvciAobGV0IGkgPSBzaGFwZS5sZW5ndGggLSAxOyBpID49IDA7IGktLSkge1xuICAgIGlmIChpbnB1dFNoYXBlW2ldID09PSBzaGFwZVtpXSkge1xuICAgICAgcmVwc1tpXSA9IDE7XG4gICAgfSBlbHNlIGlmIChpbnB1dC5zaGFwZVtpXSAhPT0gMSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBicm9hZGNhc3RUbygpOiBbJHt4U2hhcGV9XSBjYW5ub3QgYmUgYnJvYWRjYXN0IHRvIFske3NoYXBlfV0uYCk7XG4gICAgfVxuICB9XG4gIGNvbnN0IGF4ZXMgPSByZXBzLm1hcCgobiwgaSkgPT4gbiA+IDEgPyBpIDogLTEpLmZpbHRlcihpID0+IGkgPj0gMCk7XG5cbiAgaWYgKGF4ZXMubGVuZ3RoID09PSAwKSB7XG4gICAgcmV0dXJuIGNsb25lKGlucHV0KSBhcyBUZW5zb3I8Uj47XG4gIH1cblxuICAvLyBUT0RPIGNhbGwgYnJvYWRjYXN0VG8ga2VybmVsIGRpcmVjdGx5IG9uY2UgYmFja2VuZHMgaW1wbGVtZW50IGJyb2FkY3N0VG9cbiAgY29uc3QgaW5wdXRzOiBUaWxlSW5wdXRzID0ge3g6IGlucHV0fTtcbiAgY29uc3QgYXR0cnM6IFRpbGVBdHRycyA9IHtyZXBzfTtcbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBUaWxlLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGJyb2FkY2FzdFRvID0gLyogQF9fUFVSRV9fICovIG9wKHticm9hZGNhc3RUb199KTtcbiJdfQ==