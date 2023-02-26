/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { makeOnesTypedArray, sizeFromShape } from '../util';
import { assertNonNegativeIntegerDimensions } from '../util_base';
import { complex } from './complex';
import { zeros } from './zeros';
/**
 * Creates a `tf.Tensor` with all elements set to 1.
 *
 * ```js
 * tf.ones([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 *     'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function ones(shape, dtype = 'float32') {
    assertNonNegativeIntegerDimensions(shape);
    if (dtype === 'complex64') {
        const real = ones(shape, 'float32');
        const imag = zeros(shape, 'float32');
        return complex(real, imag);
    }
    const values = makeOnesTypedArray(sizeFromShape(shape), dtype);
    return ENGINE.makeTensor(values, shape, dtype);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib25lcy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL29uZXMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUdqQyxPQUFPLEVBQUMsa0JBQWtCLEVBQUUsYUFBYSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBQzFELE9BQU8sRUFBQyxrQ0FBa0MsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoRSxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFOUI7Ozs7Ozs7Ozs7OztHQVlHO0FBQ0gsTUFBTSxVQUFVLElBQUksQ0FDaEIsS0FBa0IsRUFBRSxRQUFrQixTQUFTO0lBQ2pELGtDQUFrQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzFDLElBQUksS0FBSyxLQUFLLFdBQVcsRUFBRTtRQUN6QixNQUFNLElBQUksR0FBRyxJQUFJLENBQUMsS0FBSyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQ3BDLE1BQU0sSUFBSSxHQUFHLEtBQUssQ0FBQyxLQUFLLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDckMsT0FBTyxPQUFPLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0tBQzVCO0lBQ0QsTUFBTSxNQUFNLEdBQUcsa0JBQWtCLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQy9ELE9BQU8sTUFBTSxDQUFDLFVBQVUsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLEtBQUssQ0FBYyxDQUFDO0FBQzlELENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge0RhdGFUeXBlLCBSYW5rLCBTaGFwZU1hcH0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHttYWtlT25lc1R5cGVkQXJyYXksIHNpemVGcm9tU2hhcGV9IGZyb20gJy4uL3V0aWwnO1xuaW1wb3J0IHthc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zfSBmcm9tICcuLi91dGlsX2Jhc2UnO1xuaW1wb3J0IHtjb21wbGV4fSBmcm9tICcuL2NvbXBsZXgnO1xuaW1wb3J0IHt6ZXJvc30gZnJvbSAnLi96ZXJvcyc7XG5cbi8qKlxuICogQ3JlYXRlcyBhIGB0Zi5UZW5zb3JgIHdpdGggYWxsIGVsZW1lbnRzIHNldCB0byAxLlxuICpcbiAqIGBgYGpzXG4gKiB0Zi5vbmVzKFsyLCAyXSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBzaGFwZSBBbiBhcnJheSBvZiBpbnRlZ2VycyBkZWZpbmluZyB0aGUgb3V0cHV0IHRlbnNvciBzaGFwZS5cbiAqIEBwYXJhbSBkdHlwZSBUaGUgdHlwZSBvZiBhbiBlbGVtZW50IGluIHRoZSByZXN1bHRpbmcgdGVuc29yLiBEZWZhdWx0cyB0b1xuICogICAgICdmbG9hdCcuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICovXG5leHBvcnQgZnVuY3Rpb24gb25lczxSIGV4dGVuZHMgUmFuaz4oXG4gICAgc2hhcGU6IFNoYXBlTWFwW1JdLCBkdHlwZTogRGF0YVR5cGUgPSAnZmxvYXQzMicpOiBUZW5zb3I8Uj4ge1xuICBhc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zKHNoYXBlKTtcbiAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgIGNvbnN0IHJlYWwgPSBvbmVzKHNoYXBlLCAnZmxvYXQzMicpO1xuICAgIGNvbnN0IGltYWcgPSB6ZXJvcyhzaGFwZSwgJ2Zsb2F0MzInKTtcbiAgICByZXR1cm4gY29tcGxleChyZWFsLCBpbWFnKTtcbiAgfVxuICBjb25zdCB2YWx1ZXMgPSBtYWtlT25lc1R5cGVkQXJyYXkoc2l6ZUZyb21TaGFwZShzaGFwZSksIGR0eXBlKTtcbiAgcmV0dXJuIEVOR0lORS5tYWtlVGVuc29yKHZhbHVlcywgc2hhcGUsIGR0eXBlKSBhcyBUZW5zb3I8Uj47XG59XG4iXX0=