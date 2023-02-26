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
import { getGlslDifferences } from './glsl_version';
import { useShapeUniforms } from './gpgpu_math';
import * as shader_util from './shader_compiler_util';
const CHANNEL_CHAR_TO_INDEX_MAP = {
    'R': 0,
    'G': 1,
    'B': 2,
    'A': 3
};
export class EncodeMatrixProgram {
    constructor(outputShape, inputIsUnsignedByte = false, usedChannels = 'RGBA') {
        this.variableNames = ['A'];
        this.customUniforms = [{ name: 'texShape', type: 'ivec2' }];
        const glsl = getGlslDifferences();
        this.outputShape = outputShape;
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
        let output = `result`;
        if (inputIsUnsignedByte) {
            output = `floor(result * 255. + 0.5)`;
        }
        let mainLoop = '';
        for (let usedChannelIndex = 0; usedChannelIndex < usedChannels.length; usedChannelIndex++) {
            const curChannel = usedChannels[usedChannelIndex];
            mainLoop += `
          if(offset == ${usedChannelIndex}) {
            result = values[${CHANNEL_CHAR_TO_INDEX_MAP[curChannel]}];
          }`;
        }
        this.userCode = `
      ${this.enableShapeUniforms ? shader_util.getFlatIndexFrom3DOutput() :
            shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${usedChannels.length});

        flatIndex = idiv(flatIndex, ${usedChannels.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${glsl.texture2D}(A, uv);
          ${mainLoop}
        }
        ${glsl.output} = vec4(${output}, 0., 0., 0.);
      }
    `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW5jb2RlX21hdHJpeF9ncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ2wvc3JjL2VuY29kZV9tYXRyaXhfZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ2xELE9BQU8sRUFBZSxnQkFBZ0IsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUM1RCxPQUFPLEtBQUssV0FBVyxNQUFNLHdCQUF3QixDQUFDO0FBRXRELE1BQU0seUJBQXlCLEdBQTJCO0lBQ3hELEdBQUcsRUFBRSxDQUFDO0lBQ04sR0FBRyxFQUFFLENBQUM7SUFDTixHQUFHLEVBQUUsQ0FBQztJQUNOLEdBQUcsRUFBRSxDQUFDO0NBQ1AsQ0FBQztBQUVGLE1BQU0sT0FBTyxtQkFBbUI7SUFPOUIsWUFDSSxXQUFxQyxFQUFFLG1CQUFtQixHQUFHLEtBQUssRUFDbEUsWUFBWSxHQUFHLE1BQU07UUFSekIsa0JBQWEsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBSXRCLG1CQUFjLEdBQUcsQ0FBQyxFQUFDLElBQUksRUFBRSxVQUFVLEVBQUUsSUFBSSxFQUFFLE9BQWdCLEVBQUUsQ0FBQyxDQUFDO1FBSzdELE1BQU0sSUFBSSxHQUFHLGtCQUFrQixFQUFFLENBQUM7UUFDbEMsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLG1CQUFtQixHQUFHLGdCQUFnQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUM7UUFFckUsSUFBSSxNQUFNLEdBQUcsUUFBUSxDQUFDO1FBQ3RCLElBQUksbUJBQW1CLEVBQUU7WUFDdkIsTUFBTSxHQUFHLDRCQUE0QixDQUFDO1NBQ3ZDO1FBRUQsSUFBSSxRQUFRLEdBQUcsRUFBRSxDQUFDO1FBQ2xCLEtBQUssSUFBSSxnQkFBZ0IsR0FBRyxDQUFDLEVBQUUsZ0JBQWdCLEdBQUcsWUFBWSxDQUFDLE1BQU0sRUFDaEUsZ0JBQWdCLEVBQUUsRUFBRTtZQUN2QixNQUFNLFVBQVUsR0FBRyxZQUFZLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztZQUNsRCxRQUFRLElBQUk7eUJBQ08sZ0JBQWdCOzhCQUNYLHlCQUF5QixDQUFDLFVBQVUsQ0FBQztZQUN2RCxDQUFDO1NBQ1I7UUFFRCxJQUFJLENBQUMsUUFBUSxHQUFHO1FBRVosSUFBSSxDQUFDLG1CQUFtQixDQUFDLENBQUMsQ0FBQyxXQUFXLENBQUMsd0JBQXdCLEVBQUUsQ0FBQyxDQUFDO1lBQ3hDLFdBQVcsQ0FBQyxrQkFBa0IsQ0FBQyxXQUFXLENBQUM7Ozs7Ozt1Q0FNdkMsWUFBWSxDQUFDLE1BQU07O3NDQUVwQixZQUFZLENBQUMsTUFBTTs7Ozs7OzBCQU0vQixJQUFJLENBQUMsU0FBUztZQUM1QixRQUFROztVQUVWLElBQUksQ0FBQyxNQUFNLFdBQVcsTUFBTTs7S0FFakMsQ0FBQztJQUNKLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRHbHNsRGlmZmVyZW5jZXN9IGZyb20gJy4vZ2xzbF92ZXJzaW9uJztcbmltcG9ydCB7R1BHUFVQcm9ncmFtLCB1c2VTaGFwZVVuaWZvcm1zfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0ICogYXMgc2hhZGVyX3V0aWwgZnJvbSAnLi9zaGFkZXJfY29tcGlsZXJfdXRpbCc7XG5cbmNvbnN0IENIQU5ORUxfQ0hBUl9UT19JTkRFWF9NQVA6IFJlY29yZDxzdHJpbmcsIG51bWJlcj4gPSB7XG4gICdSJzogMCxcbiAgJ0cnOiAxLFxuICAnQic6IDIsXG4gICdBJzogM1xufTtcblxuZXhwb3J0IGNsYXNzIEVuY29kZU1hdHJpeFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydBJ107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbjtcbiAgY3VzdG9tVW5pZm9ybXMgPSBbe25hbWU6ICd0ZXhTaGFwZScsIHR5cGU6ICdpdmVjMicgYXMgY29uc3QgfV07XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBvdXRwdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBpbnB1dElzVW5zaWduZWRCeXRlID0gZmFsc2UsXG4gICAgICB1c2VkQ2hhbm5lbHMgPSAnUkdCQScpIHtcbiAgICBjb25zdCBnbHNsID0gZ2V0R2xzbERpZmZlcmVuY2VzKCk7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZW5hYmxlU2hhcGVVbmlmb3JtcyA9IHVzZVNoYXBlVW5pZm9ybXModGhpcy5vdXRwdXRTaGFwZS5sZW5ndGgpO1xuXG4gICAgbGV0IG91dHB1dCA9IGByZXN1bHRgO1xuICAgIGlmIChpbnB1dElzVW5zaWduZWRCeXRlKSB7XG4gICAgICBvdXRwdXQgPSBgZmxvb3IocmVzdWx0ICogMjU1LiArIDAuNSlgO1xuICAgIH1cblxuICAgIGxldCBtYWluTG9vcCA9ICcnO1xuICAgIGZvciAobGV0IHVzZWRDaGFubmVsSW5kZXggPSAwOyB1c2VkQ2hhbm5lbEluZGV4IDwgdXNlZENoYW5uZWxzLmxlbmd0aDtcbiAgICAgICAgIHVzZWRDaGFubmVsSW5kZXgrKykge1xuICAgICAgY29uc3QgY3VyQ2hhbm5lbCA9IHVzZWRDaGFubmVsc1t1c2VkQ2hhbm5lbEluZGV4XTtcbiAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICBpZihvZmZzZXQgPT0gJHt1c2VkQ2hhbm5lbEluZGV4fSkge1xuICAgICAgICAgICAgcmVzdWx0ID0gdmFsdWVzWyR7Q0hBTk5FTF9DSEFSX1RPX0lOREVYX01BUFtjdXJDaGFubmVsXX1dO1xuICAgICAgICAgIH1gO1xuICAgIH1cblxuICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICAke1xuICAgICAgICB0aGlzLmVuYWJsZVNoYXBlVW5pZm9ybXMgPyBzaGFkZXJfdXRpbC5nZXRGbGF0SW5kZXhGcm9tM0RPdXRwdXQoKSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNoYWRlcl91dGlsLmdldEZsYXRJbmRleEZyb20zRChvdXRwdXRTaGFwZSl9XG5cbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgaXZlYzMgY29vcmRzID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIGludCBmbGF0SW5kZXggPSBnZXRGbGF0SW5kZXgoY29vcmRzKTtcbiAgICAgICAgZmxvYXQgcmVzdWx0ID0gMC47XG4gICAgICAgIGludCBvZmZzZXQgPSBpbW9kKGZsYXRJbmRleCwgJHt1c2VkQ2hhbm5lbHMubGVuZ3RofSk7XG5cbiAgICAgICAgZmxhdEluZGV4ID0gaWRpdihmbGF0SW5kZXgsICR7dXNlZENoYW5uZWxzLmxlbmd0aH0sIDEuKTtcblxuICAgICAgICBpbnQgciA9IGZsYXRJbmRleCAvIHRleFNoYXBlWzFdO1xuICAgICAgICBpZiAociA8IHRleFNoYXBlWzBdKSB7XG4gICAgICAgICAgaW50IGMgPSBpbW9kKGZsYXRJbmRleCwgdGV4U2hhcGVbMV0pO1xuICAgICAgICAgIHZlYzIgdXYgPSAodmVjMihjLCByKSArIGhhbGZDUikgLyB2ZWMyKHRleFNoYXBlWzFdLCB0ZXhTaGFwZVswXSk7XG4gICAgICAgICAgdmVjNCB2YWx1ZXMgPSAke2dsc2wudGV4dHVyZTJEfShBLCB1dik7XG4gICAgICAgICAgJHttYWluTG9vcH1cbiAgICAgICAgfVxuICAgICAgICAke2dsc2wub3V0cHV0fSA9IHZlYzQoJHtvdXRwdXR9LCAwLiwgMC4sIDAuKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG59XG4iXX0=