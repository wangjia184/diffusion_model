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
import { RotateWithOffset } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
/**
 * Rotates the input image tensor counter-clockwise with an optional offset
 * center of rotation. Currently available in the CPU, WebGL, and WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param radians The amount of rotation.
 * @param fillValue The value to fill in the empty space leftover
 *     after rotation. Can be either a single grayscale value (0-255), or an
 *     array of three numbers `[red, green, blue]` specifying the red, green,
 *     and blue channels. Defaults to `0` (black).
 * @param center The center of rotation. Can be either a single value (0-1), or
 *     an array of two numbers `[centerX, centerY]`. Defaults to `0.5` (rotates
 *     the image around its center).
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function rotateWithOffset_(image, radians, fillValue = 0, center = 0.5) {
    const $image = convertToTensor(image, 'image', 'rotateWithOffset', 'float32');
    util.assert($image.rank === 4, () => 'Error in rotateWithOffset: image must be rank 4,' +
        `but got rank ${$image.rank}.`);
    const inputs = { image: $image };
    const attrs = { radians, fillValue, center };
    const res = ENGINE.runKernel(RotateWithOffset, inputs, attrs);
    return res;
}
export const rotateWithOffset = /* @__PURE__ */ op({ rotateWithOffset_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicm90YXRlX3dpdGhfb2Zmc2V0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvaW1hZ2Uvcm90YXRlX3dpdGhfb2Zmc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLGdCQUFnQixFQUFnRCxNQUFNLG9CQUFvQixDQUFDO0FBSW5HLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEtBQUssSUFBSSxNQUFNLFlBQVksQ0FBQztBQUVuQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhDOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILFNBQVMsaUJBQWlCLENBQ3RCLEtBQTBCLEVBQUUsT0FBZSxFQUMzQyxZQUE2QyxDQUFDLEVBQzlDLFNBQWtDLEdBQUc7SUFDdkMsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsa0JBQWtCLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFOUUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDakIsR0FBRyxFQUFFLENBQUMsa0RBQWtEO1FBQ3BELGdCQUFnQixNQUFNLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUV4QyxNQUFNLE1BQU0sR0FBMkIsRUFBQyxLQUFLLEVBQUUsTUFBTSxFQUFDLENBQUM7SUFDdkQsTUFBTSxLQUFLLEdBQTBCLEVBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxNQUFNLEVBQUMsQ0FBQztJQUNsRSxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUN4QixnQkFBZ0IsRUFBRSxNQUFtQyxFQUNyRCxLQUFnQyxDQUFDLENBQUM7SUFDdEMsT0FBTyxHQUFlLENBQUM7QUFDekIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGdCQUFnQixHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxpQkFBaUIsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtSb3RhdGVXaXRoT2Zmc2V0LCBSb3RhdGVXaXRoT2Zmc2V0QXR0cnMsIFJvdGF0ZVdpdGhPZmZzZXRJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4vLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yNER9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi8uLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5cbi8qKlxuICogUm90YXRlcyB0aGUgaW5wdXQgaW1hZ2UgdGVuc29yIGNvdW50ZXItY2xvY2t3aXNlIHdpdGggYW4gb3B0aW9uYWwgb2Zmc2V0XG4gKiBjZW50ZXIgb2Ygcm90YXRpb24uIEN1cnJlbnRseSBhdmFpbGFibGUgaW4gdGhlIENQVSwgV2ViR0wsIGFuZCBXQVNNIGJhY2tlbmRzLlxuICpcbiAqIEBwYXJhbSBpbWFnZSA0ZCB0ZW5zb3Igb2Ygc2hhcGUgYFtiYXRjaCwgaW1hZ2VIZWlnaHQsIGltYWdlV2lkdGgsIGRlcHRoXWAuXG4gKiBAcGFyYW0gcmFkaWFucyBUaGUgYW1vdW50IG9mIHJvdGF0aW9uLlxuICogQHBhcmFtIGZpbGxWYWx1ZSBUaGUgdmFsdWUgdG8gZmlsbCBpbiB0aGUgZW1wdHkgc3BhY2UgbGVmdG92ZXJcbiAqICAgICBhZnRlciByb3RhdGlvbi4gQ2FuIGJlIGVpdGhlciBhIHNpbmdsZSBncmF5c2NhbGUgdmFsdWUgKDAtMjU1KSwgb3IgYW5cbiAqICAgICBhcnJheSBvZiB0aHJlZSBudW1iZXJzIGBbcmVkLCBncmVlbiwgYmx1ZV1gIHNwZWNpZnlpbmcgdGhlIHJlZCwgZ3JlZW4sXG4gKiAgICAgYW5kIGJsdWUgY2hhbm5lbHMuIERlZmF1bHRzIHRvIGAwYCAoYmxhY2spLlxuICogQHBhcmFtIGNlbnRlciBUaGUgY2VudGVyIG9mIHJvdGF0aW9uLiBDYW4gYmUgZWl0aGVyIGEgc2luZ2xlIHZhbHVlICgwLTEpLCBvclxuICogICAgIGFuIGFycmF5IG9mIHR3byBudW1iZXJzIGBbY2VudGVyWCwgY2VudGVyWV1gLiBEZWZhdWx0cyB0byBgMC41YCAocm90YXRlc1xuICogICAgIHRoZSBpbWFnZSBhcm91bmQgaXRzIGNlbnRlcikuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnSW1hZ2VzJywgbmFtZXNwYWNlOiAnaW1hZ2UnfVxuICovXG5mdW5jdGlvbiByb3RhdGVXaXRoT2Zmc2V0XyhcbiAgICBpbWFnZTogVGVuc29yNER8VGVuc29yTGlrZSwgcmFkaWFuczogbnVtYmVyLFxuICAgIGZpbGxWYWx1ZTogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IDAsXG4gICAgY2VudGVyOiBudW1iZXJ8W251bWJlciwgbnVtYmVyXSA9IDAuNSk6IFRlbnNvcjREIHtcbiAgY29uc3QgJGltYWdlID0gY29udmVydFRvVGVuc29yKGltYWdlLCAnaW1hZ2UnLCAncm90YXRlV2l0aE9mZnNldCcsICdmbG9hdDMyJyk7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkaW1hZ2UucmFuayA9PT0gNCxcbiAgICAgICgpID0+ICdFcnJvciBpbiByb3RhdGVXaXRoT2Zmc2V0OiBpbWFnZSBtdXN0IGJlIHJhbmsgNCwnICtcbiAgICAgICAgICBgYnV0IGdvdCByYW5rICR7JGltYWdlLnJhbmt9LmApO1xuXG4gIGNvbnN0IGlucHV0czogUm90YXRlV2l0aE9mZnNldElucHV0cyA9IHtpbWFnZTogJGltYWdlfTtcbiAgY29uc3QgYXR0cnM6IFJvdGF0ZVdpdGhPZmZzZXRBdHRycyA9IHtyYWRpYW5zLCBmaWxsVmFsdWUsIGNlbnRlcn07XG4gIGNvbnN0IHJlcyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBSb3RhdGVXaXRoT2Zmc2V0LCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbiAgcmV0dXJuIHJlcyBhcyBUZW5zb3I0RDtcbn1cblxuZXhwb3J0IGNvbnN0IHJvdGF0ZVdpdGhPZmZzZXQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3JvdGF0ZVdpdGhPZmZzZXRffSk7XG4iXX0=