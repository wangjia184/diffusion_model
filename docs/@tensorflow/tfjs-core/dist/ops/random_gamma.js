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
import { assertNonNegativeIntegerDimensions } from '../util_base';
import { buffer } from './buffer';
import { op } from './operation';
import { RandGamma } from './rand_util';
/**
 * Creates a `tf.Tensor` with values sampled from a gamma distribution.
 *
 * ```js
 * tf.randomGamma([2, 2], 1).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param alpha The shape parameter of the gamma distribution.
 * @param beta The inverse scale parameter of the gamma distribution. Defaults
 *     to 1.
 * @param dtype The data type of the output. Defaults to float32.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomGamma_(shape, alpha, beta = 1, dtype = 'float32', seed) {
    assertNonNegativeIntegerDimensions(shape);
    if (beta == null) {
        beta = 1;
    }
    if (dtype == null) {
        dtype = 'float32';
    }
    if (dtype !== 'float32' && dtype !== 'int32') {
        throw new Error(`Unsupported data type ${dtype}`);
    }
    const rgamma = new RandGamma(alpha, beta, dtype, seed);
    const res = buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
        res.values[i] = rgamma.nextValue();
    }
    return res.toTensor();
}
export const randomGamma = /* @__PURE__ */ op({ randomGamma_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZG9tX2dhbW1hLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcmFuZG9tX2dhbW1hLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE9BQU8sRUFBQyxrQ0FBa0MsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoRSxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUV0Qzs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxTQUFTLFlBQVksQ0FDakIsS0FBa0IsRUFBRSxLQUFhLEVBQUUsSUFBSSxHQUFHLENBQUMsRUFDM0MsUUFBMkIsU0FBUyxFQUFFLElBQWE7SUFDckQsa0NBQWtDLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDMUMsSUFBSSxJQUFJLElBQUksSUFBSSxFQUFFO1FBQ2hCLElBQUksR0FBRyxDQUFDLENBQUM7S0FDVjtJQUNELElBQUksS0FBSyxJQUFJLElBQUksRUFBRTtRQUNqQixLQUFLLEdBQUcsU0FBUyxDQUFDO0tBQ25CO0lBQ0QsSUFBSSxLQUFLLEtBQUssU0FBUyxJQUFJLEtBQUssS0FBSyxPQUFPLEVBQUU7UUFDNUMsTUFBTSxJQUFJLEtBQUssQ0FBQyx5QkFBeUIsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUNuRDtJQUNELE1BQU0sTUFBTSxHQUFHLElBQUksU0FBUyxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3ZELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDakMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQzFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLFNBQVMsRUFBRSxDQUFDO0tBQ3BDO0lBQ0QsT0FBTyxHQUFHLENBQUMsUUFBUSxFQUFFLENBQUM7QUFDeEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFdBQVcsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsWUFBWSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge1JhbmssIFNoYXBlTWFwfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydE5vbk5lZ2F0aXZlSW50ZWdlckRpbWVuc2lvbnN9IGZyb20gJy4uL3V0aWxfYmFzZSc7XG5cbmltcG9ydCB7YnVmZmVyfSBmcm9tICcuL2J1ZmZlcic7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge1JhbmRHYW1tYX0gZnJvbSAnLi9yYW5kX3V0aWwnO1xuXG4vKipcbiAqIENyZWF0ZXMgYSBgdGYuVGVuc29yYCB3aXRoIHZhbHVlcyBzYW1wbGVkIGZyb20gYSBnYW1tYSBkaXN0cmlidXRpb24uXG4gKlxuICogYGBganNcbiAqIHRmLnJhbmRvbUdhbW1hKFsyLCAyXSwgMSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBzaGFwZSBBbiBhcnJheSBvZiBpbnRlZ2VycyBkZWZpbmluZyB0aGUgb3V0cHV0IHRlbnNvciBzaGFwZS5cbiAqIEBwYXJhbSBhbHBoYSBUaGUgc2hhcGUgcGFyYW1ldGVyIG9mIHRoZSBnYW1tYSBkaXN0cmlidXRpb24uXG4gKiBAcGFyYW0gYmV0YSBUaGUgaW52ZXJzZSBzY2FsZSBwYXJhbWV0ZXIgb2YgdGhlIGdhbW1hIGRpc3RyaWJ1dGlvbi4gRGVmYXVsdHNcbiAqICAgICB0byAxLlxuICogQHBhcmFtIGR0eXBlIFRoZSBkYXRhIHR5cGUgb2YgdGhlIG91dHB1dC4gRGVmYXVsdHMgdG8gZmxvYXQzMi5cbiAqIEBwYXJhbSBzZWVkIFRoZSBzZWVkIGZvciB0aGUgcmFuZG9tIG51bWJlciBnZW5lcmF0b3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnUmFuZG9tJ31cbiAqL1xuZnVuY3Rpb24gcmFuZG9tR2FtbWFfPFIgZXh0ZW5kcyBSYW5rPihcbiAgICBzaGFwZTogU2hhcGVNYXBbUl0sIGFscGhhOiBudW1iZXIsIGJldGEgPSAxLFxuICAgIGR0eXBlOiAnZmxvYXQzMid8J2ludDMyJyA9ICdmbG9hdDMyJywgc2VlZD86IG51bWJlcik6IFRlbnNvcjxSPiB7XG4gIGFzc2VydE5vbk5lZ2F0aXZlSW50ZWdlckRpbWVuc2lvbnMoc2hhcGUpO1xuICBpZiAoYmV0YSA9PSBudWxsKSB7XG4gICAgYmV0YSA9IDE7XG4gIH1cbiAgaWYgKGR0eXBlID09IG51bGwpIHtcbiAgICBkdHlwZSA9ICdmbG9hdDMyJztcbiAgfVxuICBpZiAoZHR5cGUgIT09ICdmbG9hdDMyJyAmJiBkdHlwZSAhPT0gJ2ludDMyJykge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5zdXBwb3J0ZWQgZGF0YSB0eXBlICR7ZHR5cGV9YCk7XG4gIH1cbiAgY29uc3QgcmdhbW1hID0gbmV3IFJhbmRHYW1tYShhbHBoYSwgYmV0YSwgZHR5cGUsIHNlZWQpO1xuICBjb25zdCByZXMgPSBidWZmZXIoc2hhcGUsIGR0eXBlKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXMudmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgcmVzLnZhbHVlc1tpXSA9IHJnYW1tYS5uZXh0VmFsdWUoKTtcbiAgfVxuICByZXR1cm4gcmVzLnRvVGVuc29yKCk7XG59XG5cbmV4cG9ydCBjb25zdCByYW5kb21HYW1tYSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7cmFuZG9tR2FtbWFffSk7XG4iXX0=