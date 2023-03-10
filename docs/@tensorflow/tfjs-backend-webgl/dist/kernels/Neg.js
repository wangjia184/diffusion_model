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
import { env, Neg } from '@tensorflow/tfjs-core';
import { negImplCPU } from '../kernel_utils/shared';
import { CHECK_NAN_SNIPPET, UnaryOpProgram } from '../unaryop_gpu';
import { UnaryOpPackedProgram } from '../unaryop_packed_gpu';
const NEG = CHECK_NAN_SNIPPET + `
  return -x;
`;
const NEG_PACKED = `
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;
// This doesn't use unaryKernelFunc because negImplCPU is not of type
// SimpleUnaryKernelImplCPU.
export function neg(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (backend.shouldExecuteOnCPU([x])) {
        const xData = backend.texData.get(x.dataId);
        const [outValues, newShape] = negImplCPU(xData.values, x.shape, x.dtype);
        return backend.makeTensorInfo(newShape, x.dtype, outValues);
    }
    let program;
    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
        program = new UnaryOpPackedProgram(x.shape, NEG_PACKED);
    }
    else {
        program = new UnaryOpProgram(x.shape, NEG);
    }
    return backend.runWebGLProgram(program, [x], x.dtype);
}
export const negConfig = {
    kernelName: Neg,
    backendName: 'webgl',
    kernelFunc: neg
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTmVnLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL05lZy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUE0QixHQUFHLEVBQW9DLE1BQU0sdUJBQXVCLENBQUM7QUFHNUcsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQ2xELE9BQU8sRUFBQyxpQkFBaUIsRUFBRSxjQUFjLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUNqRSxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUUzRCxNQUFNLEdBQUcsR0FBRyxpQkFBaUIsR0FBRzs7Q0FFL0IsQ0FBQztBQUVGLE1BQU0sVUFBVSxHQUFHOzs7Ozs7Ozs7O0NBVWxCLENBQUM7QUFFRixxRUFBcUU7QUFDckUsNEJBQTRCO0FBQzVCLE1BQU0sVUFBVSxHQUFHLENBQUMsSUFBb0Q7SUFFdEUsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDL0IsTUFBTSxFQUFDLENBQUMsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUVuQixJQUFJLE9BQU8sQ0FBQyxrQkFBa0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUU7UUFDbkMsTUFBTSxLQUFLLEdBQUcsT0FBTyxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQzVDLE1BQU0sQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLEdBQ3ZCLFVBQVUsQ0FBQyxLQUFLLENBQUMsTUFBb0IsRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUM3RCxPQUFPLE9BQU8sQ0FBQyxjQUFjLENBQUMsUUFBUSxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsU0FBUyxDQUFDLENBQUM7S0FDN0Q7SUFFRCxJQUFJLE9BQTRDLENBQUM7SUFDakQsSUFBSSxHQUFHLEVBQUUsQ0FBQyxPQUFPLENBQUMsNkJBQTZCLENBQUMsRUFBRTtRQUNoRCxPQUFPLEdBQUcsSUFBSSxvQkFBb0IsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLFVBQVUsQ0FBQyxDQUFDO0tBQ3pEO1NBQU07UUFDTCxPQUFPLEdBQUcsSUFBSSxjQUFjLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxHQUFHLENBQUMsQ0FBQztLQUM1QztJQUVELE9BQU8sT0FBTyxDQUFDLGVBQWUsQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDeEQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBaUI7SUFDckMsVUFBVSxFQUFFLEdBQUc7SUFDZixXQUFXLEVBQUUsT0FBTztJQUNwQixVQUFVLEVBQUUsR0FBNEI7Q0FDekMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtlbnYsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgTmVnLCBOZWdJbnB1dHMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXl9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRXZWJHTH0gZnJvbSAnLi4vYmFja2VuZF93ZWJnbCc7XG5pbXBvcnQge25lZ0ltcGxDUFV9IGZyb20gJy4uL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuaW1wb3J0IHtDSEVDS19OQU5fU05JUFBFVCwgVW5hcnlPcFByb2dyYW19IGZyb20gJy4uL3VuYXJ5b3BfZ3B1JztcbmltcG9ydCB7VW5hcnlPcFBhY2tlZFByb2dyYW19IGZyb20gJy4uL3VuYXJ5b3BfcGFja2VkX2dwdSc7XG5cbmNvbnN0IE5FRyA9IENIRUNLX05BTl9TTklQUEVUICsgYFxuICByZXR1cm4gLXg7XG5gO1xuXG5jb25zdCBORUdfUEFDS0VEID0gYFxuICB2ZWM0IHJlc3VsdCA9IC14O1xuICBidmVjNCBpc05hTiA9IGlzbmFuKHgpO1xuXG4gIHJlc3VsdC5yID0gaXNOYU4uciA/IHguciA6IHJlc3VsdC5yO1xuICByZXN1bHQuZyA9IGlzTmFOLmcgPyB4LmcgOiByZXN1bHQuZztcbiAgcmVzdWx0LmIgPSBpc05hTi5iID8geC5iIDogcmVzdWx0LmI7XG4gIHJlc3VsdC5hID0gaXNOYU4uYSA/IHguYSA6IHJlc3VsdC5hO1xuXG4gIHJldHVybiByZXN1bHQ7XG5gO1xuXG4vLyBUaGlzIGRvZXNuJ3QgdXNlIHVuYXJ5S2VybmVsRnVuYyBiZWNhdXNlIG5lZ0ltcGxDUFUgaXMgbm90IG9mIHR5cGVcbi8vIFNpbXBsZVVuYXJ5S2VybmVsSW1wbENQVS5cbmV4cG9ydCBmdW5jdGlvbiBuZWcoYXJnczoge2lucHV0czogTmVnSW5wdXRzLCBiYWNrZW5kOiBNYXRoQmFja2VuZFdlYkdMfSk6XG4gICAgVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmR9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuXG4gIGlmIChiYWNrZW5kLnNob3VsZEV4ZWN1dGVPbkNQVShbeF0pKSB7XG4gICAgY29uc3QgeERhdGEgPSBiYWNrZW5kLnRleERhdGEuZ2V0KHguZGF0YUlkKTtcbiAgICBjb25zdCBbb3V0VmFsdWVzLCBuZXdTaGFwZV0gPVxuICAgICAgICBuZWdJbXBsQ1BVKHhEYXRhLnZhbHVlcyBhcyBUeXBlZEFycmF5LCB4LnNoYXBlLCB4LmR0eXBlKTtcbiAgICByZXR1cm4gYmFja2VuZC5tYWtlVGVuc29ySW5mbyhuZXdTaGFwZSwgeC5kdHlwZSwgb3V0VmFsdWVzKTtcbiAgfVxuXG4gIGxldCBwcm9ncmFtOiBVbmFyeU9wUHJvZ3JhbXxVbmFyeU9wUGFja2VkUHJvZ3JhbTtcbiAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX1BBQ0tfVU5BUllfT1BFUkFUSU9OUycpKSB7XG4gICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUGFja2VkUHJvZ3JhbSh4LnNoYXBlLCBORUdfUEFDS0VEKTtcbiAgfSBlbHNlIHtcbiAgICBwcm9ncmFtID0gbmV3IFVuYXJ5T3BQcm9ncmFtKHguc2hhcGUsIE5FRyk7XG4gIH1cblxuICByZXR1cm4gYmFja2VuZC5ydW5XZWJHTFByb2dyYW0ocHJvZ3JhbSwgW3hdLCB4LmR0eXBlKTtcbn1cblxuZXhwb3J0IGNvbnN0IG5lZ0NvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBOZWcsXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBuZWcgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19