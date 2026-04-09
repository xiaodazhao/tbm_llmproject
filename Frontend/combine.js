import fs from 'fs';
import path from 'path';

// 获取当前运行目录
const __dirname = process.cwd();

// 最终生成的文件名
const outputFile = path.join(__dirname, 'all_in_one_code.txt');
let outputContent = '';

// 你真正需要关心的核心目录和配置文件
const targets = ['src', 'index.html', 'vite.config.js', 'package.json'];

// 允许读取的文件后缀（过滤掉图片、字体等无关文件）
const allowedExts = ['.js', '.jsx', '.ts', '.tsx', '.vue', '.css', '.scss', '.html', '.json'];

function processTarget(targetPath) {
    if (!fs.existsSync(targetPath)) return;

    const stat = fs.statSync(targetPath);
    if (stat.isDirectory()) {
        const files = fs.readdirSync(targetPath);
        files.forEach(file => {
            processTarget(path.join(targetPath, file));
        });
    } else {
        const ext = path.extname(targetPath);
        // 只读取我们指定的代码文件
        if (allowedExts.includes(ext) || ext === '') {
            const content = fs.readFileSync(targetPath, 'utf-8');
            outputContent += `\n/* =========================================\n`;
            outputContent += `   File: ${targetPath.replace(__dirname + path.sep, '')}\n`;
            outputContent += `   ========================================= */\n\n`;
            outputContent += content + '\n';
        }
    }
}

// 开始遍历
targets.forEach(target => {
    processTarget(path.join(__dirname, target));
});

// 写入文件
fs.writeFileSync(outputFile, outputContent);
console.log(`🎉 搞定！代码已经全部打包到 ${outputFile} 里了。`);