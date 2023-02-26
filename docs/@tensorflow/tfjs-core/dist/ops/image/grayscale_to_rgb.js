/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
import { tile } from '../tile';
/**
 * Converts images from grayscale to RGB format.
 *
 * @param image A grayscale tensor to convert. The `image`'s last dimension must
 *     be size 1 with at least a two-dimensional shape.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function grayscaleToRGB_(image) {
    const $image = convertToTensor(image, 'image', 'grayscaleToRGB');
    const lastDimsIdx = $image.rank - 1;
    const lastDims = $image.shape[lastDimsIdx];
    util.assert($image.rank >= 2, () => 'Error in grayscaleToRGB: images must be at least rank 2, ' +
        `but got rank ${$image.rank}.`);
    util.assert(lastDims === 1, () => 'Error in grayscaleToRGB: last dimension of a grayscale image ' +
        `should be size 1, but got size ${lastDims}.`);
    const reps = new Array($image.rank);
    reps.fill(1, 0, lastDimsIdx);
    reps[lastDimsIdx] = 3;
    return tile($image, reps);
}
export const grayscaleToRGB = /* @__PURE__ */ op({ grayscaleToRGB_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3JheXNjYWxlX3RvX3JnYi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2ltYWdlL2dyYXlzY2FsZV90b19yZ2IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXRELE9BQU8sS0FBSyxJQUFJLE1BQU0sWUFBWSxDQUFDO0FBRW5DLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDaEMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUU3Qjs7Ozs7OztHQU9HO0FBQ0gsU0FBUyxlQUFlLENBQ1csS0FBbUI7SUFDcEQsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztJQUVqRSxNQUFNLFdBQVcsR0FBRyxNQUFNLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNwQyxNQUFNLFFBQVEsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxDQUFDO0lBRTNDLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksSUFBSSxDQUFDLEVBQ2hCLEdBQUcsRUFBRSxDQUFDLDJEQUEyRDtRQUM3RCxnQkFBZ0IsTUFBTSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFFeEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLEtBQUssQ0FBQyxFQUNkLEdBQUcsRUFBRSxDQUFDLCtEQUErRDtRQUNqRSxrQ0FBa0MsUUFBUSxHQUFHLENBQUMsQ0FBQztJQUV2RCxNQUFNLElBQUksR0FBRyxJQUFJLEtBQUssQ0FBQyxNQUFNLENBQUMsSUFBSSxDQUFDLENBQUM7SUFFcEMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQzdCLElBQUksQ0FBQyxXQUFXLENBQUMsR0FBRyxDQUFDLENBQUM7SUFFdEIsT0FBTyxJQUFJLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxDQUFDO0FBQzVCLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxjQUFjLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGVBQWUsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7VGVuc29yMkQsIFRlbnNvcjNELCBUZW5zb3I0RCwgVGVuc29yNUQsIFRlbnNvcjZEfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3RpbGV9IGZyb20gJy4uL3RpbGUnO1xuXG4vKipcbiAqIENvbnZlcnRzIGltYWdlcyBmcm9tIGdyYXlzY2FsZSB0byBSR0IgZm9ybWF0LlxuICpcbiAqIEBwYXJhbSBpbWFnZSBBIGdyYXlzY2FsZSB0ZW5zb3IgdG8gY29udmVydC4gVGhlIGBpbWFnZWAncyBsYXN0IGRpbWVuc2lvbiBtdXN0XG4gKiAgICAgYmUgc2l6ZSAxIHdpdGggYXQgbGVhc3QgYSB0d28tZGltZW5zaW9uYWwgc2hhcGUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnSW1hZ2VzJywgbmFtZXNwYWNlOiAnaW1hZ2UnfVxuICovXG5mdW5jdGlvbiBncmF5c2NhbGVUb1JHQl88VCBleHRlbmRzIFRlbnNvcjJEfFRlbnNvcjNEfFRlbnNvcjREfFRlbnNvcjVEfFxuICAgICAgICAgICAgICAgICAgICAgICAgIFRlbnNvcjZEPihpbWFnZTogVHxUZW5zb3JMaWtlKTogVCB7XG4gIGNvbnN0ICRpbWFnZSA9IGNvbnZlcnRUb1RlbnNvcihpbWFnZSwgJ2ltYWdlJywgJ2dyYXlzY2FsZVRvUkdCJyk7XG5cbiAgY29uc3QgbGFzdERpbXNJZHggPSAkaW1hZ2UucmFuayAtIDE7XG4gIGNvbnN0IGxhc3REaW1zID0gJGltYWdlLnNoYXBlW2xhc3REaW1zSWR4XTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICRpbWFnZS5yYW5rID49IDIsXG4gICAgICAoKSA9PiAnRXJyb3IgaW4gZ3JheXNjYWxlVG9SR0I6IGltYWdlcyBtdXN0IGJlIGF0IGxlYXN0IHJhbmsgMiwgJyArXG4gICAgICAgICAgYGJ1dCBnb3QgcmFuayAkeyRpbWFnZS5yYW5rfS5gKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgIGxhc3REaW1zID09PSAxLFxuICAgICAgKCkgPT4gJ0Vycm9yIGluIGdyYXlzY2FsZVRvUkdCOiBsYXN0IGRpbWVuc2lvbiBvZiBhIGdyYXlzY2FsZSBpbWFnZSAnICtcbiAgICAgICAgICBgc2hvdWxkIGJlIHNpemUgMSwgYnV0IGdvdCBzaXplICR7bGFzdERpbXN9LmApO1xuXG4gIGNvbnN0IHJlcHMgPSBuZXcgQXJyYXkoJGltYWdlLnJhbmspO1xuXG4gIHJlcHMuZmlsbCgxLCAwLCBsYXN0RGltc0lkeCk7XG4gIHJlcHNbbGFzdERpbXNJZHhdID0gMztcblxuICByZXR1cm4gdGlsZSgkaW1hZ2UsIHJlcHMpO1xufVxuXG5leHBvcnQgY29uc3QgZ3JheXNjYWxlVG9SR0IgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2dyYXlzY2FsZVRvUkdCX30pO1xuIl19