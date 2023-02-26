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
import { ENGINE } from '../../engine';
import { FFT } from '../../kernel_names';
import { assert } from '../../util';
import { op } from '../operation';
/**
 * Fast Fourier transform.
 *
 * Computes the 1-dimensional discrete Fourier transform over the inner-most
 * dimension of input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([1, 2, 3]);
 * const x = tf.complex(real, imag);
 *
 * x.fft().print();  // tf.spectral.fft(x).print();
 * ```
 * @param input The complex input to compute an fft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function fft_(input) {
    assert(input.dtype === 'complex64', () => `The dtype for tf.spectral.fft() must be complex64 ` +
        `but got ${input.dtype}.`);
    const inputs = { input };
    return ENGINE.runKernel(FFT, inputs);
}
export const fft = /* @__PURE__ */ op({ fft_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZmZ0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc3BlY3RyYWwvZmZ0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLEdBQUcsRUFBWSxNQUFNLG9CQUFvQixDQUFDO0FBR2xELE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7Ozs7Ozs7Ozs7OztHQWdCRztBQUNILFNBQVMsSUFBSSxDQUFDLEtBQWE7SUFDekIsTUFBTSxDQUNGLEtBQUssQ0FBQyxLQUFLLEtBQUssV0FBVyxFQUMzQixHQUFHLEVBQUUsQ0FBQyxvREFBb0Q7UUFDdEQsV0FBVyxLQUFLLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQztJQUVuQyxNQUFNLE1BQU0sR0FBYyxFQUFDLEtBQUssRUFBQyxDQUFDO0lBRWxDLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxHQUFHLEVBQUUsTUFBbUMsQ0FBQyxDQUFDO0FBQ3BFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxHQUFHLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLElBQUksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtGRlQsIEZGVElucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7YXNzZXJ0fSBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5cbi8qKlxuICogRmFzdCBGb3VyaWVyIHRyYW5zZm9ybS5cbiAqXG4gKiBDb21wdXRlcyB0aGUgMS1kaW1lbnNpb25hbCBkaXNjcmV0ZSBGb3VyaWVyIHRyYW5zZm9ybSBvdmVyIHRoZSBpbm5lci1tb3N0XG4gKiBkaW1lbnNpb24gb2YgaW5wdXQuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHJlYWwgPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICogY29uc3QgaW1hZyA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKiBjb25zdCB4ID0gdGYuY29tcGxleChyZWFsLCBpbWFnKTtcbiAqXG4gKiB4LmZmdCgpLnByaW50KCk7ICAvLyB0Zi5zcGVjdHJhbC5mZnQoeCkucHJpbnQoKTtcbiAqIGBgYFxuICogQHBhcmFtIGlucHV0IFRoZSBjb21wbGV4IGlucHV0IHRvIGNvbXB1dGUgYW4gZmZ0IG92ZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU3BlY3RyYWwnLCBuYW1lc3BhY2U6ICdzcGVjdHJhbCd9XG4gKi9cbmZ1bmN0aW9uIGZmdF8oaW5wdXQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIGFzc2VydChcbiAgICAgIGlucHV0LmR0eXBlID09PSAnY29tcGxleDY0JyxcbiAgICAgICgpID0+IGBUaGUgZHR5cGUgZm9yIHRmLnNwZWN0cmFsLmZmdCgpIG11c3QgYmUgY29tcGxleDY0IGAgK1xuICAgICAgICAgIGBidXQgZ290ICR7aW5wdXQuZHR5cGV9LmApO1xuXG4gIGNvbnN0IGlucHV0czogRkZUSW5wdXRzID0ge2lucHV0fTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChGRlQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGZmdCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7ZmZ0X30pO1xuIl19