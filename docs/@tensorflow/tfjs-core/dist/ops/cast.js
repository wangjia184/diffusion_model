/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import { Cast } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Casts a `tf.Tensor` to a new dtype.
 *
 * ```js
 * const x = tf.tensor1d([1.5, 2.5, 3]);
 * tf.cast(x, 'int32').print();
 * ```
 * @param x The input tensor to be casted.
 * @param dtype The dtype to cast the input tensor to.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function cast_(x, dtype) {
    const $x = convertToTensor(x, 'x', 'cast');
    // Sanity checks.
    if (!util.isValidDtype(dtype)) {
        throw new Error(`Failed to cast to unknown dtype ${dtype}`);
    }
    if (dtype === 'string' && $x.dtype !== 'string' ||
        dtype !== 'string' && $x.dtype === 'string') {
        throw new Error('Only strings can be casted to strings');
    }
    const inputs = { x: $x };
    const attrs = { dtype };
    return ENGINE.runKernel(Cast, inputs, attrs);
}
export const cast = /* @__PURE__ */ op({ cast_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY2FzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2Nhc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsSUFBSSxFQUF3QixNQUFNLGlCQUFpQixDQUFDO0FBSTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7OztHQVdHO0FBQ0gsU0FBUyxLQUFLLENBQW1CLENBQWUsRUFBRSxLQUFlO0lBQy9ELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBRTNDLGlCQUFpQjtJQUNqQixJQUFJLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxLQUFLLENBQUMsRUFBRTtRQUM3QixNQUFNLElBQUksS0FBSyxDQUFDLG1DQUFtQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzdEO0lBQ0QsSUFBSSxLQUFLLEtBQUssUUFBUSxJQUFJLEVBQUUsQ0FBQyxLQUFLLEtBQUssUUFBUTtRQUMzQyxLQUFLLEtBQUssUUFBUSxJQUFJLEVBQUUsQ0FBQyxLQUFLLEtBQUssUUFBUSxFQUFFO1FBQy9DLE1BQU0sSUFBSSxLQUFLLENBQUMsdUNBQXVDLENBQUMsQ0FBQztLQUMxRDtJQUVELE1BQU0sTUFBTSxHQUFlLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ25DLE1BQU0sS0FBSyxHQUFjLEVBQUMsS0FBSyxFQUFDLENBQUM7SUFFakMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixJQUFJLEVBQUUsTUFBbUMsRUFDekMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxJQUFJLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLEtBQUssRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7Q2FzdCwgQ2FzdEF0dHJzLCBDYXN0SW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7RGF0YVR5cGUsIFRlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDYXN0cyBhIGB0Zi5UZW5zb3JgIHRvIGEgbmV3IGR0eXBlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEuNSwgMi41LCAzXSk7XG4gKiB0Zi5jYXN0KHgsICdpbnQzMicpLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IgdG8gYmUgY2FzdGVkLlxuICogQHBhcmFtIGR0eXBlIFRoZSBkdHlwZSB0byBjYXN0IHRoZSBpbnB1dCB0ZW5zb3IgdG8uXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnVHJhbnNmb3JtYXRpb25zJ31cbiAqL1xuZnVuY3Rpb24gY2FzdF88VCBleHRlbmRzIFRlbnNvcj4oeDogVHxUZW5zb3JMaWtlLCBkdHlwZTogRGF0YVR5cGUpOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnY2FzdCcpO1xuXG4gIC8vIFNhbml0eSBjaGVja3MuXG4gIGlmICghdXRpbC5pc1ZhbGlkRHR5cGUoZHR5cGUpKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBGYWlsZWQgdG8gY2FzdCB0byB1bmtub3duIGR0eXBlICR7ZHR5cGV9YCk7XG4gIH1cbiAgaWYgKGR0eXBlID09PSAnc3RyaW5nJyAmJiAkeC5kdHlwZSAhPT0gJ3N0cmluZycgfHxcbiAgICAgIGR0eXBlICE9PSAnc3RyaW5nJyAmJiAkeC5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ09ubHkgc3RyaW5ncyBjYW4gYmUgY2FzdGVkIHRvIHN0cmluZ3MnKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogQ2FzdElucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBDYXN0QXR0cnMgPSB7ZHR5cGV9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQ2FzdCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBjYXN0ID0gLyogQF9fUFVSRV9fICovIG9wKHtjYXN0X30pO1xuIl19