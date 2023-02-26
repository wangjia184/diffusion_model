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
import { ENGINE } from '../../engine';
import { StringSplit } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * Split elements of `input` based on `delimiter` into a SparseTensor .
 *
 * Let N be the size of source (typically N will be the batch size). Split each
 * element of `input` based on `delimiter` and return a SparseTensor containing
 * the splitted tokens. Empty tokens are ignored if `skipEmpty` is set to True.
 *
 * `delimiter` can be empty, or a string of split characters. If `delimiter` is
 * an empty string, each element of `input` is split into individual
 * character strings. Otherwise every character of `delimiter` is a potential
 * split point.
 *
 * ```js
 * const result = tf.string.stringSplit(['hello world',  'a b c'], ' ');
 * result['indices'].print(); // [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
 * result['values'].print(); // ['hello', 'world', 'a', 'b', 'c']
 * result['shape'].print(); // [2, 3]
 * ```
 * @param input: 1-D. Strings to split.
 * @param delimiter: 0-D. Delimiter characters, or empty string.
 * @param skipEmpty: Optional. If true, skip the empty strings from the result.
 *     Defaults to true.
 * @return A map with the following properties:
 *     - indices: A dense matrix of int32 representing the indices of the sparse
 *       tensor.
 *     - values: A vector of strings corresponding to the splited values.
 *     - shape: a length-2 vector of int32 representing the shape of the sparse
 * tensor, where the first value is N and the second value is the maximum number
 * of tokens in a single input entry.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
function stringSplit_(input, delimiter, skipEmpty = true) {
    const $input = convertToTensor(input, 'input', 'stringSplit', 'string');
    const $delimiter = convertToTensor(delimiter, 'delimiter', 'stringSplit', 'string');
    if ($input.rank !== 1) {
        throw new Error(`Input should be Tensor1D but received shape ${$input.shape}`);
    }
    if ($delimiter.rank !== 0) {
        throw new Error(`Delimiter should be a scalar but received shape ${$delimiter.shape}`);
    }
    const attrs = { skipEmpty };
    const inputs = { input: $input, delimiter: $delimiter };
    const result = ENGINE.runKernel(StringSplit, inputs, attrs);
    return { indices: result[0], values: result[1], shape: result[2] };
}
export const stringSplit = /* @__PURE__ */ op({ stringSplit_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RyaW5nX3NwbGl0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc3RyaW5nL3N0cmluZ19zcGxpdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxXQUFXLEVBQXNDLE1BQU0sb0JBQW9CLENBQUM7QUFHcEYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXRELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFaEM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0ErQkc7QUFDSCxTQUFTLFlBQVksQ0FDakIsS0FBMEIsRUFBRSxTQUE0QixFQUN4RCxTQUFTLEdBQUcsSUFBSTtJQUNsQixNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDeEUsTUFBTSxVQUFVLEdBQ1osZUFBZSxDQUFDLFNBQVMsRUFBRSxXQUFXLEVBQUUsYUFBYSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBRXJFLElBQUksTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDckIsTUFBTSxJQUFJLEtBQUssQ0FDWCwrQ0FBK0MsTUFBTSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDcEU7SUFDRCxJQUFJLFVBQVUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ3pCLE1BQU0sSUFBSSxLQUFLLENBQ1gsbURBQW1ELFVBQVUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzVFO0lBRUQsTUFBTSxLQUFLLEdBQXFCLEVBQUMsU0FBUyxFQUFDLENBQUM7SUFDNUMsTUFBTSxNQUFNLEdBQXNCLEVBQUMsS0FBSyxFQUFFLE1BQU0sRUFBRSxTQUFTLEVBQUUsVUFBVSxFQUFDLENBQUM7SUFDekUsTUFBTSxNQUFNLEdBQ1IsTUFBTSxDQUFDLFNBQVMsQ0FBQyxXQUFXLEVBQUUsTUFBWSxFQUFFLEtBQVcsQ0FBQyxDQUFDO0lBQzdELE9BQU8sRUFBQyxPQUFPLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDO0FBQ25FLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtTdHJpbmdTcGxpdCwgU3RyaW5nU3BsaXRBdHRycywgU3RyaW5nU3BsaXRJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1NjYWxhciwgVGVuc29yLCBUZW5zb3IxRH0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7U2NhbGFyTGlrZSwgVGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBTcGxpdCBlbGVtZW50cyBvZiBgaW5wdXRgIGJhc2VkIG9uIGBkZWxpbWl0ZXJgIGludG8gYSBTcGFyc2VUZW5zb3IgLlxuICpcbiAqIExldCBOIGJlIHRoZSBzaXplIG9mIHNvdXJjZSAodHlwaWNhbGx5IE4gd2lsbCBiZSB0aGUgYmF0Y2ggc2l6ZSkuIFNwbGl0IGVhY2hcbiAqIGVsZW1lbnQgb2YgYGlucHV0YCBiYXNlZCBvbiBgZGVsaW1pdGVyYCBhbmQgcmV0dXJuIGEgU3BhcnNlVGVuc29yIGNvbnRhaW5pbmdcbiAqIHRoZSBzcGxpdHRlZCB0b2tlbnMuIEVtcHR5IHRva2VucyBhcmUgaWdub3JlZCBpZiBgc2tpcEVtcHR5YCBpcyBzZXQgdG8gVHJ1ZS5cbiAqXG4gKiBgZGVsaW1pdGVyYCBjYW4gYmUgZW1wdHksIG9yIGEgc3RyaW5nIG9mIHNwbGl0IGNoYXJhY3RlcnMuIElmIGBkZWxpbWl0ZXJgIGlzXG4gKiBhbiBlbXB0eSBzdHJpbmcsIGVhY2ggZWxlbWVudCBvZiBgaW5wdXRgIGlzIHNwbGl0IGludG8gaW5kaXZpZHVhbFxuICogY2hhcmFjdGVyIHN0cmluZ3MuIE90aGVyd2lzZSBldmVyeSBjaGFyYWN0ZXIgb2YgYGRlbGltaXRlcmAgaXMgYSBwb3RlbnRpYWxcbiAqIHNwbGl0IHBvaW50LlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCByZXN1bHQgPSB0Zi5zdHJpbmcuc3RyaW5nU3BsaXQoWydoZWxsbyB3b3JsZCcsICAnYSBiIGMnXSwgJyAnKTtcbiAqIHJlc3VsdFsnaW5kaWNlcyddLnByaW50KCk7IC8vIFtbMCwgMF0sIFswLCAxXSwgWzEsIDBdLCBbMSwgMV0sIFsxLCAyXV1cbiAqIHJlc3VsdFsndmFsdWVzJ10ucHJpbnQoKTsgLy8gWydoZWxsbycsICd3b3JsZCcsICdhJywgJ2InLCAnYyddXG4gKiByZXN1bHRbJ3NoYXBlJ10ucHJpbnQoKTsgLy8gWzIsIDNdXG4gKiBgYGBcbiAqIEBwYXJhbSBpbnB1dDogMS1ELiBTdHJpbmdzIHRvIHNwbGl0LlxuICogQHBhcmFtIGRlbGltaXRlcjogMC1ELiBEZWxpbWl0ZXIgY2hhcmFjdGVycywgb3IgZW1wdHkgc3RyaW5nLlxuICogQHBhcmFtIHNraXBFbXB0eTogT3B0aW9uYWwuIElmIHRydWUsIHNraXAgdGhlIGVtcHR5IHN0cmluZ3MgZnJvbSB0aGUgcmVzdWx0LlxuICogICAgIERlZmF1bHRzIHRvIHRydWUuXG4gKiBAcmV0dXJuIEEgbWFwIHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogICAgIC0gaW5kaWNlczogQSBkZW5zZSBtYXRyaXggb2YgaW50MzIgcmVwcmVzZW50aW5nIHRoZSBpbmRpY2VzIG9mIHRoZSBzcGFyc2VcbiAqICAgICAgIHRlbnNvci5cbiAqICAgICAtIHZhbHVlczogQSB2ZWN0b3Igb2Ygc3RyaW5ncyBjb3JyZXNwb25kaW5nIHRvIHRoZSBzcGxpdGVkIHZhbHVlcy5cbiAqICAgICAtIHNoYXBlOiBhIGxlbmd0aC0yIHZlY3RvciBvZiBpbnQzMiByZXByZXNlbnRpbmcgdGhlIHNoYXBlIG9mIHRoZSBzcGFyc2VcbiAqIHRlbnNvciwgd2hlcmUgdGhlIGZpcnN0IHZhbHVlIGlzIE4gYW5kIHRoZSBzZWNvbmQgdmFsdWUgaXMgdGhlIG1heGltdW0gbnVtYmVyXG4gKiBvZiB0b2tlbnMgaW4gYSBzaW5nbGUgaW5wdXQgZW50cnkuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU3RyaW5nJ31cbiAqL1xuZnVuY3Rpb24gc3RyaW5nU3BsaXRfKFxuICAgIGlucHV0OiBUZW5zb3IxRHxUZW5zb3JMaWtlLCBkZWxpbWl0ZXI6IFNjYWxhcnxTY2FsYXJMaWtlLFxuICAgIHNraXBFbXB0eSA9IHRydWUpOiBOYW1lZFRlbnNvck1hcCB7XG4gIGNvbnN0ICRpbnB1dCA9IGNvbnZlcnRUb1RlbnNvcihpbnB1dCwgJ2lucHV0JywgJ3N0cmluZ1NwbGl0JywgJ3N0cmluZycpO1xuICBjb25zdCAkZGVsaW1pdGVyID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihkZWxpbWl0ZXIsICdkZWxpbWl0ZXInLCAnc3RyaW5nU3BsaXQnLCAnc3RyaW5nJyk7XG5cbiAgaWYgKCRpbnB1dC5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgSW5wdXQgc2hvdWxkIGJlIFRlbnNvcjFEIGJ1dCByZWNlaXZlZCBzaGFwZSAkeyRpbnB1dC5zaGFwZX1gKTtcbiAgfVxuICBpZiAoJGRlbGltaXRlci5yYW5rICE9PSAwKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgRGVsaW1pdGVyIHNob3VsZCBiZSBhIHNjYWxhciBidXQgcmVjZWl2ZWQgc2hhcGUgJHskZGVsaW1pdGVyLnNoYXBlfWApO1xuICB9XG5cbiAgY29uc3QgYXR0cnM6IFN0cmluZ1NwbGl0QXR0cnMgPSB7c2tpcEVtcHR5fTtcbiAgY29uc3QgaW5wdXRzOiBTdHJpbmdTcGxpdElucHV0cyA9IHtpbnB1dDogJGlucHV0LCBkZWxpbWl0ZXI6ICRkZWxpbWl0ZXJ9O1xuICBjb25zdCByZXN1bHQ6IFRlbnNvcltdID1cbiAgICAgIEVOR0lORS5ydW5LZXJuZWwoU3RyaW5nU3BsaXQsIGlucHV0cyBhcyB7fSwgYXR0cnMgYXMge30pO1xuICByZXR1cm4ge2luZGljZXM6IHJlc3VsdFswXSwgdmFsdWVzOiByZXN1bHRbMV0sIHNoYXBlOiByZXN1bHRbMl19O1xufVxuXG5leHBvcnQgY29uc3Qgc3RyaW5nU3BsaXQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3N0cmluZ1NwbGl0X30pO1xuIl19