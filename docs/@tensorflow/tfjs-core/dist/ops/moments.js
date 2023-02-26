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
import { convertToTensor } from '../tensor_util_env';
import { parseAxisParam } from '../util';
import { expandShapeToKeepDim } from './axis_util';
import { cast } from './cast';
import { mean } from './mean';
import { op } from './operation';
import { reshape } from './reshape';
import { square } from './square';
import { sub } from './sub';
/**
 * Calculates the mean and variance of `x`. The mean and variance are
 * calculated by aggregating the contents of `x` across `axes`. If `x` is
 * 1-D and `axes = [0]` this is just the mean and variance of a vector.
 *
 * @param x The input tensor.
 * @param axis The dimension(s) along with to compute mean and
 *     variance. By default it reduces all dimensions.
 * @param keepDims If true, the moments have the same dimensionality as the
 *     input.
 * @return An object with two keys: `mean` and `variance`.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function moments_(x, axis = null, keepDims = false) {
    x = convertToTensor(x, 'x', 'moments');
    const axes = parseAxisParam(axis, x.shape);
    const xMean = mean(x, axes, keepDims);
    let keepDimsShape = xMean.shape;
    if (!keepDims) {
        keepDimsShape = expandShapeToKeepDim(xMean.shape, axes);
    }
    const devSquared = square(sub(cast(x, 'float32'), reshape(xMean, keepDimsShape)));
    const variance = mean(devSquared, axes, keepDims);
    return { mean: xMean, variance };
}
export const moments = /* @__PURE__ */ op({ moments_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibW9tZW50cy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL21vbWVudHMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdkMsT0FBTyxFQUFDLG9CQUFvQixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQ2pELE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDNUIsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUM1QixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUNoQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBRTFCOzs7Ozs7Ozs7Ozs7O0dBYUc7QUFDSCxTQUFTLFFBQVEsQ0FDYixDQUFvQixFQUFFLE9BQXdCLElBQUksRUFDbEQsUUFBUSxHQUFHLEtBQUs7SUFDbEIsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQ3ZDLE1BQU0sSUFBSSxHQUFHLGNBQWMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzNDLE1BQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ3RDLElBQUksYUFBYSxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUM7SUFDaEMsSUFBSSxDQUFDLFFBQVEsRUFBRTtRQUNiLGFBQWEsR0FBRyxvQkFBb0IsQ0FBQyxLQUFLLENBQUMsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO0tBQ3pEO0lBQ0QsTUFBTSxVQUFVLEdBQ1osTUFBTSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxLQUFLLEVBQUUsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ25FLE1BQU0sUUFBUSxHQUFHLElBQUksQ0FBQyxVQUFVLEVBQUUsSUFBSSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ2xELE9BQU8sRUFBQyxJQUFJLEVBQUUsS0FBSyxFQUFFLFFBQVEsRUFBQyxDQUFDO0FBQ2pDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFFBQVEsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7cGFyc2VBeGlzUGFyYW19IGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2V4cGFuZFNoYXBlVG9LZWVwRGltfSBmcm9tICcuL2F4aXNfdXRpbCc7XG5pbXBvcnQge2Nhc3R9IGZyb20gJy4vY2FzdCc7XG5pbXBvcnQge21lYW59IGZyb20gJy4vbWVhbic7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5pbXBvcnQge3NxdWFyZX0gZnJvbSAnLi9zcXVhcmUnO1xuaW1wb3J0IHtzdWJ9IGZyb20gJy4vc3ViJztcblxuLyoqXG4gKiBDYWxjdWxhdGVzIHRoZSBtZWFuIGFuZCB2YXJpYW5jZSBvZiBgeGAuIFRoZSBtZWFuIGFuZCB2YXJpYW5jZSBhcmVcbiAqIGNhbGN1bGF0ZWQgYnkgYWdncmVnYXRpbmcgdGhlIGNvbnRlbnRzIG9mIGB4YCBhY3Jvc3MgYGF4ZXNgLiBJZiBgeGAgaXNcbiAqIDEtRCBhbmQgYGF4ZXMgPSBbMF1gIHRoaXMgaXMganVzdCB0aGUgbWVhbiBhbmQgdmFyaWFuY2Ugb2YgYSB2ZWN0b3IuXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvci5cbiAqIEBwYXJhbSBheGlzIFRoZSBkaW1lbnNpb24ocykgYWxvbmcgd2l0aCB0byBjb21wdXRlIG1lYW4gYW5kXG4gKiAgICAgdmFyaWFuY2UuIEJ5IGRlZmF1bHQgaXQgcmVkdWNlcyBhbGwgZGltZW5zaW9ucy5cbiAqIEBwYXJhbSBrZWVwRGltcyBJZiB0cnVlLCB0aGUgbW9tZW50cyBoYXZlIHRoZSBzYW1lIGRpbWVuc2lvbmFsaXR5IGFzIHRoZVxuICogICAgIGlucHV0LlxuICogQHJldHVybiBBbiBvYmplY3Qgd2l0aCB0d28ga2V5czogYG1lYW5gIGFuZCBgdmFyaWFuY2VgLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ05vcm1hbGl6YXRpb24nfVxuICovXG5mdW5jdGlvbiBtb21lbnRzXyhcbiAgICB4OiBUZW5zb3J8VGVuc29yTGlrZSwgYXhpczogbnVtYmVyfG51bWJlcltdID0gbnVsbCxcbiAgICBrZWVwRGltcyA9IGZhbHNlKToge21lYW46IFRlbnNvciwgdmFyaWFuY2U6IFRlbnNvcn0ge1xuICB4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ21vbWVudHMnKTtcbiAgY29uc3QgYXhlcyA9IHBhcnNlQXhpc1BhcmFtKGF4aXMsIHguc2hhcGUpO1xuICBjb25zdCB4TWVhbiA9IG1lYW4oeCwgYXhlcywga2VlcERpbXMpO1xuICBsZXQga2VlcERpbXNTaGFwZSA9IHhNZWFuLnNoYXBlO1xuICBpZiAoIWtlZXBEaW1zKSB7XG4gICAga2VlcERpbXNTaGFwZSA9IGV4cGFuZFNoYXBlVG9LZWVwRGltKHhNZWFuLnNoYXBlLCBheGVzKTtcbiAgfVxuICBjb25zdCBkZXZTcXVhcmVkID1cbiAgICAgIHNxdWFyZShzdWIoY2FzdCh4LCAnZmxvYXQzMicpLCByZXNoYXBlKHhNZWFuLCBrZWVwRGltc1NoYXBlKSkpO1xuICBjb25zdCB2YXJpYW5jZSA9IG1lYW4oZGV2U3F1YXJlZCwgYXhlcywga2VlcERpbXMpO1xuICByZXR1cm4ge21lYW46IHhNZWFuLCB2YXJpYW5jZX07XG59XG5cbmV4cG9ydCBjb25zdCBtb21lbnRzID0gLyogQF9fUFVSRV9fICovIG9wKHttb21lbnRzX30pO1xuIl19