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
import { MaxPoolWithArgmax } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the 2D max pooling of an image with Argmax index.
 * The indices in argmax are flattened, so that a maximum value at position `[b,
 * y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if
 * include_batch_in_index is False; `((b * height + y) * width + x) * channels
 * +c` if include_batch_in_index is True.
 *
 * The indices returned are always in `[0, height) x [0, width)` before
 * flattening.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param includeBatchIndex Defaults to False. Whether to include batch
 *    dimension in flattened index of argmax.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function maxPoolWithArgmax_(x, filterSize, strides, pad, includeBatchInIndex = false) {
    const $x = convertToTensor(x, 'x', 'maxPoolWithArgmax');
    const inputs = { x: $x };
    const attrs = { filterSize, strides, pad, includeBatchInIndex };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = ENGINE.runKernel(MaxPoolWithArgmax, inputs, attrs);
    return { result: result[0], indexes: result[1] };
}
export const maxPoolWithArgmax = /* @__PURE__ */ op({ maxPoolWithArgmax_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF4X3Bvb2xfd2l0aF9hcmdtYXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9tYXhfcG9vbF93aXRoX2FyZ21heC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxpQkFBaUIsRUFBa0QsTUFBTSxpQkFBaUIsQ0FBQztBQUluRyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQ0c7QUFDSCxTQUFTLGtCQUFrQixDQUN2QixDQUFlLEVBQUUsVUFBbUMsRUFDcEQsT0FBZ0MsRUFBRSxHQUEwQixFQUM1RCxtQkFBbUIsR0FBRyxLQUFLO0lBQzdCLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLG1CQUFtQixDQUFDLENBQUM7SUFFeEQsTUFBTSxNQUFNLEdBQTRCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ2hELE1BQU0sS0FBSyxHQUNrQixFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFFLG1CQUFtQixFQUFDLENBQUM7SUFFN0UsMERBQTBEO0lBQzFELE1BQU0sTUFBTSxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1osaUJBQWlCLEVBQUUsTUFBbUMsRUFDdEQsS0FBZ0MsQ0FBYSxDQUFDO0lBRWpFLE9BQU8sRUFBQyxNQUFNLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUMsQ0FBQztBQUNqRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0saUJBQWlCLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGtCQUFrQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge01heFBvb2xXaXRoQXJnbWF4LCBNYXhQb29sV2l0aEFyZ21heEF0dHJzLCBNYXhQb29sV2l0aEFyZ21heElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjREfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgMkQgbWF4IHBvb2xpbmcgb2YgYW4gaW1hZ2Ugd2l0aCBBcmdtYXggaW5kZXguXG4gKiBUaGUgaW5kaWNlcyBpbiBhcmdtYXggYXJlIGZsYXR0ZW5lZCwgc28gdGhhdCBhIG1heGltdW0gdmFsdWUgYXQgcG9zaXRpb24gYFtiLFxuICogeSwgeCwgY11gIGJlY29tZXMgZmxhdHRlbmVkIGluZGV4OiBgKHkgKiB3aWR0aCArIHgpICogY2hhbm5lbHMgKyBjYCBpZlxuICogaW5jbHVkZV9iYXRjaF9pbl9pbmRleCBpcyBGYWxzZTsgYCgoYiAqIGhlaWdodCArIHkpICogd2lkdGggKyB4KSAqIGNoYW5uZWxzXG4gKiArY2AgaWYgaW5jbHVkZV9iYXRjaF9pbl9pbmRleCBpcyBUcnVlLlxuICpcbiAqIFRoZSBpbmRpY2VzIHJldHVybmVkIGFyZSBhbHdheXMgaW4gYFswLCBoZWlnaHQpIHggWzAsIHdpZHRoKWAgYmVmb3JlXG4gKiBmbGF0dGVuaW5nLlxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IsIG9mIHJhbmsgNCBvciByYW5rIDMgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpcyBhc3N1bWVkLlxuICogQHBhcmFtIGZpbHRlclNpemUgVGhlIGZpbHRlciBzaXplOiBgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGhdYC4gSWZcbiAqICAgICBgZmlsdGVyU2l6ZWAgaXMgYSBzaW5nbGUgbnVtYmVyLCB0aGVuIGBmaWx0ZXJIZWlnaHQgPT0gZmlsdGVyV2lkdGhgLlxuICogQHBhcmFtIHN0cmlkZXMgVGhlIHN0cmlkZXMgb2YgdGhlIHBvb2xpbmc6IGBbc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aF1gLiBJZlxuICogICAgIGBzdHJpZGVzYCBpcyBhIHNpbmdsZSBudW1iZXIsIHRoZW4gYHN0cmlkZUhlaWdodCA9PSBzdHJpZGVXaWR0aGAuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdCBBbiBvcHRpb25hbCBzdHJpbmcgZnJvbTogXCJOREhXQ1wiLCBcIk5DREhXXCIuIERlZmF1bHRzIHRvXG4gKiAgICAgXCJOREhXQ1wiLiBTcGVjaWZ5IHRoZSBkYXRhIGZvcm1hdCBvZiB0aGUgaW5wdXQgYW5kIG91dHB1dCBkYXRhLiBXaXRoIHRoZVxuICogICAgIGRlZmF1bHQgZm9ybWF0IFwiTkRIV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgIGRlcHRoLCBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc10uIE9ubHkgXCJOREhXQ1wiIGlzIGN1cnJlbnRseSBzdXBwb3J0ZWQuXG4gKiBAcGFyYW0gcGFkIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtLlxuICogICAgLSBgc2FtZWAgYW5kIHN0cmlkZSAxOiBvdXRwdXQgd2lsbCBiZSBvZiBzYW1lIHNpemUgYXMgaW5wdXQsXG4gKiAgICAgICByZWdhcmRsZXNzIG9mIGZpbHRlciBzaXplLlxuICogICAgLSBgdmFsaWRgOiBvdXRwdXQgd2lsbCBiZSBzbWFsbGVyIHRoYW4gaW5wdXQgaWYgZmlsdGVyIGlzIGxhcmdlclxuICogICAgICAgdGhhbiAxeDEuXG4gKiAgICAtIEZvciBtb3JlIGluZm8sIHNlZSB0aGlzIGd1aWRlOlxuICogICAgIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb25dKFxuICogICAgICAgICAgaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uKVxuICogQHBhcmFtIGluY2x1ZGVCYXRjaEluZGV4IERlZmF1bHRzIHRvIEZhbHNlLiBXaGV0aGVyIHRvIGluY2x1ZGUgYmF0Y2hcbiAqICAgIGRpbWVuc2lvbiBpbiBmbGF0dGVuZWQgaW5kZXggb2YgYXJnbWF4LlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uJ31cbiAqL1xuZnVuY3Rpb24gbWF4UG9vbFdpdGhBcmdtYXhfPFQgZXh0ZW5kcyBUZW5zb3I0RD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBmaWx0ZXJTaXplOiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdfG51bWJlciwgcGFkOiAndmFsaWQnfCdzYW1lJ3xudW1iZXIsXG4gICAgaW5jbHVkZUJhdGNoSW5JbmRleCA9IGZhbHNlKTogTmFtZWRUZW5zb3JNYXAge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdtYXhQb29sV2l0aEFyZ21heCcpO1xuXG4gIGNvbnN0IGlucHV0czogTWF4UG9vbFdpdGhBcmdtYXhJbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczpcbiAgICAgIE1heFBvb2xXaXRoQXJnbWF4QXR0cnMgPSB7ZmlsdGVyU2l6ZSwgc3RyaWRlcywgcGFkLCBpbmNsdWRlQmF0Y2hJbkluZGV4fTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIGNvbnN0IHJlc3VsdCA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICAgICBNYXhQb29sV2l0aEFyZ21heCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCkgYXMgVGVuc29yW107XG5cbiAgcmV0dXJuIHtyZXN1bHQ6IHJlc3VsdFswXSwgaW5kZXhlczogcmVzdWx0WzFdfTtcbn1cblxuZXhwb3J0IGNvbnN0IG1heFBvb2xXaXRoQXJnbWF4ID0gLyogQF9fUFVSRV9fICovIG9wKHttYXhQb29sV2l0aEFyZ21heF99KTtcbiJdfQ==