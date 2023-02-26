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
import { ENGINE } from '../engine';
import { BroadcastArgs } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Return the shape of s0 op s1 with broadcast.
 *
 * compute r0, the broadcasted shape as a tensor.
 * s0, s1 and r0 are all integer vectors.
 *
 * This function returns the shape of the result of an operation between
 * two tensors of size s0 and s1 performed with broadcast.
 *
 * @param s0 A tensor representing a shape
 * @param s1 A tensor representing a shape
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function broadcastArgs_(s0, s1) {
    const shape1Input = convertToTensor(s0, 's0', 'broadcastArgs', 'int32');
    const shape2Input = convertToTensor(s1, 's1', 'broadcastArgs', 'int32');
    if (shape1Input.rank !== 1) {
        throw new Error('broadcastArgs(): first input must be a vector (rank=1). ' +
            `Has rank ${shape1Input.rank}`);
    }
    if (shape2Input.rank !== 1) {
        throw new Error('broadcastArgs(): second input must be a vector (rank=1). ' +
            `Has rank ${shape2Input.rank}`);
    }
    const inputs = { s0: shape1Input, s1: shape2Input };
    return ENGINE.runKernel(BroadcastArgs, inputs);
}
export const broadcastArgs = /* @__PURE__ */ op({ broadcastArgs_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYnJvYWRjYXN0X2FyZ3MuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9icm9hZGNhc3RfYXJncy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUUsTUFBTSxFQUFFLE1BQU0sV0FBVyxDQUFDO0FBQ25DLE9BQU8sRUFBRSxhQUFhLEVBQXVCLE1BQU0saUJBQWlCLENBQUM7QUFFckUsT0FBTyxFQUFFLGVBQWUsRUFBRSxNQUFNLG9CQUFvQixDQUFDO0FBR3JELE9BQU8sRUFBRSxFQUFFLEVBQUUsTUFBTSxhQUFhLENBQUM7QUFFakM7Ozs7Ozs7Ozs7Ozs7R0FhRztBQUNILFNBQVMsY0FBYyxDQUNyQixFQUF1QixFQUFFLEVBQXVCO0lBQ2hELE1BQU0sV0FBVyxHQUFHLGVBQWUsQ0FBQyxFQUFFLEVBQUUsSUFBSSxFQUFFLGVBQWUsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUN4RSxNQUFNLFdBQVcsR0FBRyxlQUFlLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxlQUFlLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFFeEUsSUFBSSxXQUFXLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUMxQixNQUFNLElBQUksS0FBSyxDQUNiLDBEQUEwRDtZQUMxRCxZQUFZLFdBQVcsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDO0tBQ25DO0lBRUQsSUFBSSxXQUFXLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUMxQixNQUFNLElBQUksS0FBSyxDQUNiLDJEQUEyRDtZQUMzRCxZQUFZLFdBQVcsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDO0tBQ25DO0lBRUQsTUFBTSxNQUFNLEdBQXdCLEVBQUUsRUFBRSxFQUFFLFdBQVcsRUFBRSxFQUFFLEVBQUUsV0FBVyxFQUFFLENBQUM7SUFDekUsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLGFBQWEsRUFBRSxNQUFtQyxDQUFDLENBQUM7QUFDOUUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUUsY0FBYyxFQUFFLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHsgTmFtZWRUZW5zb3JNYXAgfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHsgRU5HSU5FIH0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7IEJyb2FkY2FzdEFyZ3MsIEJyb2FkY2FzdEFyZ3NJbnB1dHMgfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHsgVGVuc29yIH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7IGNvbnZlcnRUb1RlbnNvciB9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQgeyBSYW5rLCBUZW5zb3JMaWtlIH0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQgeyBvcCB9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBSZXR1cm4gdGhlIHNoYXBlIG9mIHMwIG9wIHMxIHdpdGggYnJvYWRjYXN0LlxuICpcbiAqIGNvbXB1dGUgcjAsIHRoZSBicm9hZGNhc3RlZCBzaGFwZSBhcyBhIHRlbnNvci5cbiAqIHMwLCBzMSBhbmQgcjAgYXJlIGFsbCBpbnRlZ2VyIHZlY3RvcnMuXG4gKlxuICogVGhpcyBmdW5jdGlvbiByZXR1cm5zIHRoZSBzaGFwZSBvZiB0aGUgcmVzdWx0IG9mIGFuIG9wZXJhdGlvbiBiZXR3ZWVuXG4gKiB0d28gdGVuc29ycyBvZiBzaXplIHMwIGFuZCBzMSBwZXJmb3JtZWQgd2l0aCBicm9hZGNhc3QuXG4gKlxuICogQHBhcmFtIHMwIEEgdGVuc29yIHJlcHJlc2VudGluZyBhIHNoYXBlXG4gKiBAcGFyYW0gczEgQSB0ZW5zb3IgcmVwcmVzZW50aW5nIGEgc2hhcGVcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdUcmFuc2Zvcm1hdGlvbnMnfVxuICovXG5mdW5jdGlvbiBicm9hZGNhc3RBcmdzXzxSIGV4dGVuZHMgUmFuaz4oXG4gIHMwOiBUZW5zb3IgfCBUZW5zb3JMaWtlLCBzMTogVGVuc29yIHwgVGVuc29yTGlrZSk6IFRlbnNvcjxSPiB7XG4gIGNvbnN0IHNoYXBlMUlucHV0ID0gY29udmVydFRvVGVuc29yKHMwLCAnczAnLCAnYnJvYWRjYXN0QXJncycsICdpbnQzMicpO1xuICBjb25zdCBzaGFwZTJJbnB1dCA9IGNvbnZlcnRUb1RlbnNvcihzMSwgJ3MxJywgJ2Jyb2FkY2FzdEFyZ3MnLCAnaW50MzInKTtcblxuICBpZiAoc2hhcGUxSW5wdXQucmFuayAhPT0gMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICdicm9hZGNhc3RBcmdzKCk6IGZpcnN0IGlucHV0IG11c3QgYmUgYSB2ZWN0b3IgKHJhbms9MSkuICcgK1xuICAgICAgYEhhcyByYW5rICR7c2hhcGUxSW5wdXQucmFua31gKTtcbiAgfVxuXG4gIGlmIChzaGFwZTJJbnB1dC5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgJ2Jyb2FkY2FzdEFyZ3MoKTogc2Vjb25kIGlucHV0IG11c3QgYmUgYSB2ZWN0b3IgKHJhbms9MSkuICcgK1xuICAgICAgYEhhcyByYW5rICR7c2hhcGUySW5wdXQucmFua31gKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogQnJvYWRjYXN0QXJnc0lucHV0cyA9IHsgczA6IHNoYXBlMUlucHV0LCBzMTogc2hhcGUySW5wdXQgfTtcbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoQnJvYWRjYXN0QXJncywgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXApO1xufVxuXG5leHBvcnQgY29uc3QgYnJvYWRjYXN0QXJncyA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7IGJyb2FkY2FzdEFyZ3NfIH0pO1xuIl19