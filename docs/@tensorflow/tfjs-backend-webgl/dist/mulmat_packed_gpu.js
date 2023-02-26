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
import { useShapeUniforms } from './gpgpu_math';
export class MatMulPackedProgram {
    constructor(aShape, bShape, outputShape, transposeA = false, transposeB = false, addBias = false, activation = null, hasPreluActivation = false, hasLeakyreluActivation = false) {
        this.variableNames = ['matrixA', 'matrixB'];
        this.packedInputs = true;
        this.packedOutput = true;
        this.outputShape = outputShape;
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
        const sharedDim = transposeA ? aShape[1] : aShape[2];
        const sharedDimensionPacked = Math.ceil(sharedDim / 2);
        const aSample = transposeA ? 'i * 2, rc.y' : 'rc.y, i * 2';
        const bSample = transposeB ? 'rc.z, i * 2' : 'i * 2, rc.z';
        const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
        const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];
        let activationSnippet = '', applyActivationSnippet = '';
        if (activation) {
            if (hasPreluActivation) {
                activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
            }
            else if (hasLeakyreluActivation) {
                activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activation}
        }`;
            }
            else {
                activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
            }
            applyActivationSnippet = `result = activation(result);`;
        }
        const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        if (hasLeakyreluActivation) {
            this.variableNames.push('leakyreluAlpha');
        }
        let batchASnippet = 'rc.x';
        let batchBSnippet = 'rc.x';
        if (aShape[0] < bShape[0]) {
            batchASnippet = `imod(rc.x, ${aShape[0]})`;
        }
        else if (bShape[0] < aShape[0]) {
            batchBSnippet = `imod(rc.x, ${bShape[0]})`;
        }
        this.userCode = `
      ${activationSnippet}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${sharedDimensionPacked}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${batchASnippet};
        int batchB = ${batchBSnippet};
        for (int i = 0; i < ${sharedDimensionPacked}; i++) {
          vec4 a = getMatrixA(batchA, ${aSample});
          vec4 b = getMatrixB(batchB, ${bSample});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${aSwizzle[0]} * ${bSwizzle[0]});
          result += (${aSwizzle[1]} * ${bSwizzle[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${addBiasSnippet}

        ${applyActivationSnippet}

        setOutput(result);
      }
    `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibXVsbWF0X3BhY2tlZF9ncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ2wvc3JjL211bG1hdF9wYWNrZWRfZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBZSxnQkFBZ0IsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUU1RCxNQUFNLE9BQU8sbUJBQW1CO0lBUTlCLFlBQ0ksTUFBZ0MsRUFBRSxNQUFnQyxFQUNsRSxXQUFxQyxFQUFFLFVBQVUsR0FBRyxLQUFLLEVBQ3pELFVBQVUsR0FBRyxLQUFLLEVBQUUsT0FBTyxHQUFHLEtBQUssRUFBRSxhQUFxQixJQUFJLEVBQzlELGtCQUFrQixHQUFHLEtBQUssRUFBRSxzQkFBc0IsR0FBRyxLQUFLO1FBWDlELGtCQUFhLEdBQUcsQ0FBQyxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDdkMsaUJBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsaUJBQVksR0FBRyxJQUFJLENBQUM7UUFVbEIsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLG1CQUFtQixHQUFHLGdCQUFnQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUM7UUFFckUsTUFBTSxTQUFTLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNyRCxNQUFNLHFCQUFxQixHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsU0FBUyxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBRXZELE1BQU0sT0FBTyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxhQUFhLENBQUM7UUFDM0QsTUFBTSxPQUFPLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLGFBQWEsQ0FBQztRQUMzRCxNQUFNLFFBQVEsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsUUFBUSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsRUFBRSxRQUFRLENBQUMsQ0FBQztRQUMxRSxNQUFNLFFBQVEsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsUUFBUSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsRUFBRSxRQUFRLENBQUMsQ0FBQztRQUUxRSxJQUFJLGlCQUFpQixHQUFHLEVBQUUsRUFBRSxzQkFBc0IsR0FBRyxFQUFFLENBQUM7UUFDeEQsSUFBSSxVQUFVLEVBQUU7WUFDZCxJQUFJLGtCQUFrQixFQUFFO2dCQUN0QixpQkFBaUIsR0FBRzs7WUFFaEIsVUFBVTtVQUNaLENBQUM7YUFDSjtpQkFBTSxJQUFJLHNCQUFzQixFQUFFO2dCQUNqQyxpQkFBaUIsR0FBRzs7WUFFaEIsVUFBVTtVQUNaLENBQUM7YUFDSjtpQkFBTTtnQkFDTCxpQkFBaUIsR0FBRztZQUNoQixVQUFVO1VBQ1osQ0FBQzthQUNKO1lBRUQsc0JBQXNCLEdBQUcsOEJBQThCLENBQUM7U0FDekQ7UUFFRCxNQUFNLGNBQWMsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLGlDQUFpQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7UUFDeEUsSUFBSSxPQUFPLEVBQUU7WUFDWCxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztTQUNqQztRQUVELElBQUksa0JBQWtCLEVBQUU7WUFDdEIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQztTQUNuRDtRQUVELElBQUksc0JBQXNCLEVBQUU7WUFDMUIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztTQUMzQztRQUVELElBQUksYUFBYSxHQUFHLE1BQU0sQ0FBQztRQUMzQixJQUFJLGFBQWEsR0FBRyxNQUFNLENBQUM7UUFDM0IsSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFO1lBQ3pCLGFBQWEsR0FBRyxjQUFjLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDO1NBQzVDO2FBQU0sSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFO1lBQ2hDLGFBQWEsR0FBRyxjQUFjLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDO1NBQzVDO1FBRUQsSUFBSSxDQUFDLFFBQVEsR0FBRztRQUNaLGlCQUFpQjs7c0NBRWEscUJBQXFCOzs7O3VCQUlwQyxhQUFhO3VCQUNiLGFBQWE7OEJBQ04scUJBQXFCO3dDQUNYLE9BQU87d0NBQ1AsT0FBTzs7Ozt1QkFJeEIsUUFBUSxDQUFDLENBQUMsQ0FBQyxNQUFNLFFBQVEsQ0FBQyxDQUFDLENBQUM7dUJBQzVCLFFBQVEsQ0FBQyxDQUFDLENBQUMsTUFBTSxRQUFRLENBQUMsQ0FBQyxDQUFDOzs7Ozs7Ozs7VUFTekMsY0FBYzs7VUFFZCxzQkFBc0I7Ozs7S0FJM0IsQ0FBQztJQUNKLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtHUEdQVVByb2dyYW0sIHVzZVNoYXBlVW5pZm9ybXN9IGZyb20gJy4vZ3BncHVfbWF0aCc7XG5cbmV4cG9ydCBjbGFzcyBNYXRNdWxQYWNrZWRQcm9ncmFtIGltcGxlbWVudHMgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsnbWF0cml4QScsICdtYXRyaXhCJ107XG4gIHBhY2tlZElucHV0cyA9IHRydWU7XG4gIHBhY2tlZE91dHB1dCA9IHRydWU7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgdXNlckNvZGU6IHN0cmluZztcbiAgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGFTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBiU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIG91dHB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHRyYW5zcG9zZUEgPSBmYWxzZSxcbiAgICAgIHRyYW5zcG9zZUIgPSBmYWxzZSwgYWRkQmlhcyA9IGZhbHNlLCBhY3RpdmF0aW9uOiBzdHJpbmcgPSBudWxsLFxuICAgICAgaGFzUHJlbHVBY3RpdmF0aW9uID0gZmFsc2UsIGhhc0xlYWt5cmVsdUFjdGl2YXRpb24gPSBmYWxzZSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBvdXRwdXRTaGFwZTtcbiAgICB0aGlzLmVuYWJsZVNoYXBlVW5pZm9ybXMgPSB1c2VTaGFwZVVuaWZvcm1zKHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoKTtcblxuICAgIGNvbnN0IHNoYXJlZERpbSA9IHRyYW5zcG9zZUEgPyBhU2hhcGVbMV0gOiBhU2hhcGVbMl07XG4gICAgY29uc3Qgc2hhcmVkRGltZW5zaW9uUGFja2VkID0gTWF0aC5jZWlsKHNoYXJlZERpbSAvIDIpO1xuXG4gICAgY29uc3QgYVNhbXBsZSA9IHRyYW5zcG9zZUEgPyAnaSAqIDIsIHJjLnknIDogJ3JjLnksIGkgKiAyJztcbiAgICBjb25zdCBiU2FtcGxlID0gdHJhbnNwb3NlQiA/ICdyYy56LCBpICogMicgOiAnaSAqIDIsIHJjLnonO1xuICAgIGNvbnN0IGFTd2l6emxlID0gdHJhbnNwb3NlQSA/IFsnYS54eHl5JywgJ2Euenp3dyddIDogWydhLnh4enonLCAnYS55eXd3J107XG4gICAgY29uc3QgYlN3aXp6bGUgPSB0cmFuc3Bvc2VCID8gWydiLnh6eHonLCAnYi55d3l3J10gOiBbJ2IueHl4eScsICdiLnp3encnXTtcblxuICAgIGxldCBhY3RpdmF0aW9uU25pcHBldCA9ICcnLCBhcHBseUFjdGl2YXRpb25TbmlwcGV0ID0gJyc7XG4gICAgaWYgKGFjdGl2YXRpb24pIHtcbiAgICAgIGlmIChoYXNQcmVsdUFjdGl2YXRpb24pIHtcbiAgICAgICAgYWN0aXZhdGlvblNuaXBwZXQgPSBgdmVjNCBhY3RpdmF0aW9uKHZlYzQgYSkge1xuICAgICAgICAgIHZlYzQgYiA9IGdldFByZWx1QWN0aXZhdGlvbldlaWdodHNBdE91dENvb3JkcygpO1xuICAgICAgICAgICR7YWN0aXZhdGlvbn1cbiAgICAgICAgfWA7XG4gICAgICB9IGVsc2UgaWYgKGhhc0xlYWt5cmVsdUFjdGl2YXRpb24pIHtcbiAgICAgICAgYWN0aXZhdGlvblNuaXBwZXQgPSBgdmVjNCBhY3RpdmF0aW9uKHZlYzQgYSkge1xuICAgICAgICAgIHZlYzQgYiA9IGdldExlYWt5cmVsdUFscGhhQXRPdXRDb29yZHMoKTtcbiAgICAgICAgICAke2FjdGl2YXRpb259XG4gICAgICAgIH1gO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYWN0aXZhdGlvblNuaXBwZXQgPSBgdmVjNCBhY3RpdmF0aW9uKHZlYzQgeCkge1xuICAgICAgICAgICR7YWN0aXZhdGlvbn1cbiAgICAgICAgfWA7XG4gICAgICB9XG5cbiAgICAgIGFwcGx5QWN0aXZhdGlvblNuaXBwZXQgPSBgcmVzdWx0ID0gYWN0aXZhdGlvbihyZXN1bHQpO2A7XG4gICAgfVxuXG4gICAgY29uc3QgYWRkQmlhc1NuaXBwZXQgPSBhZGRCaWFzID8gJ3Jlc3VsdCArPSBnZXRCaWFzQXRPdXRDb29yZHMoKTsnIDogJyc7XG4gICAgaWYgKGFkZEJpYXMpIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdiaWFzJyk7XG4gICAgfVxuXG4gICAgaWYgKGhhc1ByZWx1QWN0aXZhdGlvbikge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ3ByZWx1QWN0aXZhdGlvbldlaWdodHMnKTtcbiAgICB9XG5cbiAgICBpZiAoaGFzTGVha3lyZWx1QWN0aXZhdGlvbikge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ2xlYWt5cmVsdUFscGhhJyk7XG4gICAgfVxuXG4gICAgbGV0IGJhdGNoQVNuaXBwZXQgPSAncmMueCc7XG4gICAgbGV0IGJhdGNoQlNuaXBwZXQgPSAncmMueCc7XG4gICAgaWYgKGFTaGFwZVswXSA8IGJTaGFwZVswXSkge1xuICAgICAgYmF0Y2hBU25pcHBldCA9IGBpbW9kKHJjLngsICR7YVNoYXBlWzBdfSlgO1xuICAgIH0gZWxzZSBpZiAoYlNoYXBlWzBdIDwgYVNoYXBlWzBdKSB7XG4gICAgICBiYXRjaEJTbmlwcGV0ID0gYGltb2QocmMueCwgJHtiU2hhcGVbMF19KWA7XG4gICAgfVxuXG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgICR7YWN0aXZhdGlvblNuaXBwZXR9XG4gICAgICAvLyBEb24ndCB1c2UgdW5pZm9ybSBmb3Igc2hhcmVkRGltZW5zaW9uUGFja2VkIGZvciBwZXJmb3JtYW5jZS5cbiAgICAgIGNvbnN0IGZsb2F0IHNoYXJlZERpbWVuc2lvbiA9ICR7c2hhcmVkRGltZW5zaW9uUGFja2VkfS4wO1xuXG4gICAgICB2ZWM0IGRvdDJ4MkFSb3dCQ29sKGl2ZWMzIHJjKSB7XG4gICAgICAgIHZlYzQgcmVzdWx0ID0gdmVjNCgwKTtcbiAgICAgICAgaW50IGJhdGNoQSA9ICR7YmF0Y2hBU25pcHBldH07XG4gICAgICAgIGludCBiYXRjaEIgPSAke2JhdGNoQlNuaXBwZXR9O1xuICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8ICR7c2hhcmVkRGltZW5zaW9uUGFja2VkfTsgaSsrKSB7XG4gICAgICAgICAgdmVjNCBhID0gZ2V0TWF0cml4QShiYXRjaEEsICR7YVNhbXBsZX0pO1xuICAgICAgICAgIHZlYzQgYiA9IGdldE1hdHJpeEIoYmF0Y2hCLCAke2JTYW1wbGV9KTtcblxuICAgICAgICAgIC8vIFRoZXNlIHN3aXp6bGVkIHByb2R1Y3RzIG5lZWQgdG8gYmUgc2VwYXJhdGVseSBhZGRlZC5cbiAgICAgICAgICAvLyBTZWU6IGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzE3MzVcbiAgICAgICAgICByZXN1bHQgKz0gKCR7YVN3aXp6bGVbMF19ICogJHtiU3dpenpsZVswXX0pO1xuICAgICAgICAgIHJlc3VsdCArPSAoJHthU3dpenpsZVsxXX0gKiAke2JTd2l6emxlWzFdfSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICAgIH1cblxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICBpdmVjMyByYyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICB2ZWM0IHJlc3VsdCA9IGRvdDJ4MkFSb3dCQ29sKHJjKTtcblxuICAgICAgICAke2FkZEJpYXNTbmlwcGV0fVxuXG4gICAgICAgICR7YXBwbHlBY3RpdmF0aW9uU25pcHBldH1cblxuICAgICAgICBzZXRPdXRwdXQocmVzdWx0KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG59XG4iXX0=