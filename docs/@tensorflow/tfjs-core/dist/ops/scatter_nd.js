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
import { ScatterNd } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { assertNonNegativeIntegerDimensions } from '../util_base';
import { op } from './operation';
import * as scatter_nd_util from './scatter_nd_util';
/**
 * Creates a new tensor by applying sparse updates to individual
 * values or slices within a zero tensor of the given shape tensor according to
 * indices. This operator is the inverse of the `tf.gatherND` operator which
 * extracts values or slices from a given tensor.
 *
 * ```js
 * const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
 * const updates = tf.tensor1d([9, 10, 11, 12]);
 * const shape = [8];
 * tf.scatterND(indices, updates, shape).print() //[0, 11, 0, 10, 9, 0, 0, 12]
 * ```
 *
 * @param indices The tensor contains the indices into the output tensor.
 * @param updates The tensor contains the value for the indices.
 * @param shape: The shape of the output tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function scatterND_(indices, updates, shape) {
    assertNonNegativeIntegerDimensions(shape);
    const $indices = convertToTensor(indices, 'indices', 'scatterND', 'int32');
    const $updates = convertToTensor(updates, 'updates', 'scatterND');
    scatter_nd_util.validateInput($updates, $indices, shape);
    const inputs = { indices: $indices, updates: $updates };
    const attrs = { shape };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(ScatterNd, inputs, attrs);
}
export const scatterND = /* @__PURE__ */ op({ scatterND_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2NhdHRlcl9uZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NjYXR0ZXJfbmQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsU0FBUyxFQUFrQyxNQUFNLGlCQUFpQixDQUFDO0FBSTNFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEVBQUMsa0NBQWtDLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFaEUsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEtBQUssZUFBZSxNQUFNLG1CQUFtQixDQUFDO0FBRXJEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FrQkc7QUFDSCxTQUFTLFVBQVUsQ0FDZixPQUEwQixFQUFFLE9BQTBCLEVBQ3RELEtBQWtCO0lBQ3BCLGtDQUFrQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzFDLE1BQU0sUUFBUSxHQUFHLGVBQWUsQ0FBQyxPQUFPLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUMzRSxNQUFNLFFBQVEsR0FBRyxlQUFlLENBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNsRSxlQUFlLENBQUMsYUFBYSxDQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFFekQsTUFBTSxNQUFNLEdBQW9CLEVBQUMsT0FBTyxFQUFFLFFBQVEsRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFDLENBQUM7SUFDdkUsTUFBTSxLQUFLLEdBQW1CLEVBQUMsS0FBSyxFQUFDLENBQUM7SUFFdEMsMERBQTBEO0lBQzFELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDWixTQUFTLEVBQUUsTUFBbUMsRUFDOUMsS0FBZ0MsQ0FBYyxDQUFDO0FBQzVELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFVBQVUsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtTY2F0dGVyTmQsIFNjYXR0ZXJOZEF0dHJzLCBTY2F0dGVyTmRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtSYW5rLCBTaGFwZU1hcCwgVGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zfSBmcm9tICcuLi91dGlsX2Jhc2UnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQgKiBhcyBzY2F0dGVyX25kX3V0aWwgZnJvbSAnLi9zY2F0dGVyX25kX3V0aWwnO1xuXG4vKipcbiAqIENyZWF0ZXMgYSBuZXcgdGVuc29yIGJ5IGFwcGx5aW5nIHNwYXJzZSB1cGRhdGVzIHRvIGluZGl2aWR1YWxcbiAqIHZhbHVlcyBvciBzbGljZXMgd2l0aGluIGEgemVybyB0ZW5zb3Igb2YgdGhlIGdpdmVuIHNoYXBlIHRlbnNvciBhY2NvcmRpbmcgdG9cbiAqIGluZGljZXMuIFRoaXMgb3BlcmF0b3IgaXMgdGhlIGludmVyc2Ugb2YgdGhlIGB0Zi5nYXRoZXJORGAgb3BlcmF0b3Igd2hpY2hcbiAqIGV4dHJhY3RzIHZhbHVlcyBvciBzbGljZXMgZnJvbSBhIGdpdmVuIHRlbnNvci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5kaWNlcyA9IHRmLnRlbnNvcjJkKFs0LCAzLCAxLCA3XSwgWzQsIDFdLCAnaW50MzInKTtcbiAqIGNvbnN0IHVwZGF0ZXMgPSB0Zi50ZW5zb3IxZChbOSwgMTAsIDExLCAxMl0pO1xuICogY29uc3Qgc2hhcGUgPSBbOF07XG4gKiB0Zi5zY2F0dGVyTkQoaW5kaWNlcywgdXBkYXRlcywgc2hhcGUpLnByaW50KCkgLy9bMCwgMTEsIDAsIDEwLCA5LCAwLCAwLCAxMl1cbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBpbmRpY2VzIFRoZSB0ZW5zb3IgY29udGFpbnMgdGhlIGluZGljZXMgaW50byB0aGUgb3V0cHV0IHRlbnNvci5cbiAqIEBwYXJhbSB1cGRhdGVzIFRoZSB0ZW5zb3IgY29udGFpbnMgdGhlIHZhbHVlIGZvciB0aGUgaW5kaWNlcy5cbiAqIEBwYXJhbSBzaGFwZTogVGhlIHNoYXBlIG9mIHRoZSBvdXRwdXQgdGVuc29yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1NsaWNpbmcgYW5kIEpvaW5pbmcnfVxuICovXG5mdW5jdGlvbiBzY2F0dGVyTkRfPFIgZXh0ZW5kcyBSYW5rPihcbiAgICBpbmRpY2VzOiBUZW5zb3J8VGVuc29yTGlrZSwgdXBkYXRlczogVGVuc29yfFRlbnNvckxpa2UsXG4gICAgc2hhcGU6IFNoYXBlTWFwW1JdKTogVGVuc29yPFI+IHtcbiAgYXNzZXJ0Tm9uTmVnYXRpdmVJbnRlZ2VyRGltZW5zaW9ucyhzaGFwZSk7XG4gIGNvbnN0ICRpbmRpY2VzID0gY29udmVydFRvVGVuc29yKGluZGljZXMsICdpbmRpY2VzJywgJ3NjYXR0ZXJORCcsICdpbnQzMicpO1xuICBjb25zdCAkdXBkYXRlcyA9IGNvbnZlcnRUb1RlbnNvcih1cGRhdGVzLCAndXBkYXRlcycsICdzY2F0dGVyTkQnKTtcbiAgc2NhdHRlcl9uZF91dGlsLnZhbGlkYXRlSW5wdXQoJHVwZGF0ZXMsICRpbmRpY2VzLCBzaGFwZSk7XG5cbiAgY29uc3QgaW5wdXRzOiBTY2F0dGVyTmRJbnB1dHMgPSB7aW5kaWNlczogJGluZGljZXMsIHVwZGF0ZXM6ICR1cGRhdGVzfTtcbiAgY29uc3QgYXR0cnM6IFNjYXR0ZXJOZEF0dHJzID0ge3NoYXBlfTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgIFNjYXR0ZXJOZCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFRlbnNvcjxSPjtcbn1cblxuZXhwb3J0IGNvbnN0IHNjYXR0ZXJORCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c2NhdHRlck5EX30pO1xuIl19